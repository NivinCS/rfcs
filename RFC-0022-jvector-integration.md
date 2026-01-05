# **RFC-0021 for Presto**

## Vector Similarity Search with JVector Integration

Proposers

* Nivin CS
* [Additional contributors]

## Related Issues

[To be created]

## Summary

This RFC proposes integrating JVector, a high-performance vector similarity search library, into Presto to enable efficient Approximate Nearest Neighbor (ANN) search capabilities. This integration will allow users to perform semantic search, similarity matching, and other vector-based operations directly within Presto SQL queries, supporting modern AI/ML workloads including embeddings from Large Language Models (LLMs) and other machine learning models.

## Background

### Motivation

Vector embeddings have become fundamental to modern AI/ML applications, including:
- **Semantic Search**: Finding similar documents, images, or other content based on meaning rather than exact matches
- **Recommendation Systems**: Identifying similar items or users based on behavioral embeddings
- **Retrieval-Augmented Generation (RAG)**: Retrieving relevant context for LLM applications
- **Anomaly Detection**: Identifying outliers in high-dimensional feature spaces
- **Image and Video Search**: Finding visually similar content

Currently, Presto lacks native support for efficient vector similarity search. Users must either:
1. Export data to specialized vector databases (adding complexity and data movement)
2. Implement inefficient brute-force similarity calculations in SQL
3. Use external services, breaking the unified query interface

This RFC proposes native vector search capabilities in Presto, enabling users to:
- Store vector embeddings alongside structured data in Iceberg tables
- Perform efficient ANN searches using SQL
- Leverage Presto's distributed architecture for scalable vector search
- Maintain data locality and avoid unnecessary data movement

### Why JVector?

JVector (https://github.com/jbellis/jvector) is chosen for the following reasons:
- **Pure Java Implementation**: Seamless integration with Presto's Java codebase
- **High Performance**: Competitive with native implementations (HNSW algorithm)
- **Apache 2.0 License**: Compatible with Presto's licensing
- **Disk-Based Indexes**: Supports large-scale datasets without memory constraints
- **Multiple Similarity Functions**: Cosine, Euclidean, Dot Product
- **Active Development**: Well-maintained with regular updates

### Use Case Example

Consider a document search system with embeddings:

```sql
-- Create table with embeddings
CREATE TABLE documents (
    doc_id BIGINT,
    title VARCHAR,
    content VARCHAR,
    embedding ARRAY(REAL),  -- 768-dimensional embedding
    created_date DATE
) WITH (
    partitioning = ARRAY['created_date']
);

-- Find top 10 most similar documents to a query
SELECT d.doc_id, d.title, ann.distance
FROM documents d
JOIN TABLE(approx_nearest_neighbors(
    query_vector => ARRAY[0.1, 0.2, ..., 0.768],
    column_name => 'iceberg.default.documents.embedding',
    limit => 10
)) ann ON d.doc_id = ann.row_id
ORDER BY ann.distance
LIMIT 10;
```

## Proposed Implementation

### Architecture Overview

The implementation follows a distributed architecture leveraging Presto's coordinator-worker model:

```
┌─────────────────────────────────────────────────────────────┐
│                        Coordinator                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Query Planning & Optimization                      │    │
│  │  - Parse TVF call                                   │    │
│  │  - Resolve vector index metadata                    │    │
│  │  - Generate splits for partitioned indexes          │    │
│  │  - Plan result aggregation                          │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Top-K Aggregator                                   │    │
│  │  - Merge results from workers                       │    │
│  │  - Maintain global top-K heap                       │    │
│  │  - Return final results                             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Distribute Splits
                            ▼
        ┌───────────────────┴───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼─────────┐
│   Worker 1     │  │   Worker 2     │  │   Worker N     │
│ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │
│ │Index Cache │ │  │ │Index Cache │ │  │ │Index Cache │ │
│ └────────────┘ │  │ └────────────┘ │  │ └────────────┘ │
│ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │
│ │ANN Search  │ │  │ │ANN Search  │ │  │ │ANN Search  │ │
│ │Partition 1 │ │  │ │Partition 2 │ │  │ │Partition N │ │
│ └────────────┘ │  │ └────────────┘ │  │ └────────────┘ │
│ Local Top-K    │  │ Local Top-K    │  │ Local Top-K    │
└────────────────┘  └────────────────┘  └────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                    Results to Coordinator
```

### Component Architecture

#### 1. Index Management Layer

**VectorIndexMetadata**
Stores metadata about vector indexes in the Iceberg metadata layer:

```java
public class VectorIndexMetadata {
    private final String indexName;
    private final String tableName;
    private final String columnName;
    private final VectorSimilarityFunction similarityFunction;
    private final int dimension;
    private final int m;  // HNSW parameter
    private final int efConstruction;  // HNSW parameter
    private final long createdTime;
    private final long lastModifiedTime;
    private final Map<String, String> partitionToIndexPath;  // partition -> S3 path
    private final IndexStatus status;  // BUILDING, READY, FAILED
    private final long totalVectors;
}
```

**VectorIndexManager**
Manages index lifecycle:

```java
public interface VectorIndexManager {
    /**
     * Create a new vector index
     */
    VectorIndexMetadata createIndex(
        ConnectorSession session,
        SchemaTableName tableName,
        String columnName,
        VectorIndexConfig config);
    
    /**
     * Get index metadata
     */
    Optional<VectorIndexMetadata> getIndexMetadata(
        ConnectorSession session,
        SchemaTableName tableName,
        String columnName);
    
    /**
     * List all indexes for a table
     */
    List<VectorIndexMetadata> listIndexes(
        ConnectorSession session,
        SchemaTableName tableName);
    
    /**
     * Drop an index
     */
    void dropIndex(
        ConnectorSession session,
        SchemaTableName tableName,
        String indexName);
    
    /**
     * Refresh/rebuild an index
     */
    void refreshIndex(
        ConnectorSession session,
        SchemaTableName tableName,
        String indexName);
}
```

#### 2. Index Building

**Distributed Index Building Strategy**

For partitioned tables, indexes are built per partition in parallel:

```java
public class DistributedVectorIndexBuilder {
    /**
     * Build indexes for all partitions in parallel
     */
    public Map<String, Path> buildPartitionedIndexes(
        ConnectorSession session,
        SchemaTableName tableName,
        String columnName,
        VectorIndexConfig config) {
        
        // 1. Get table partitions
        List<PartitionInfo> partitions = getTablePartitions(tableName);
        
        // 2. Create index building tasks for each partition
        List<IndexBuildTask> tasks = partitions.stream()
            .map(partition -> new IndexBuildTask(
                partition,
                columnName,
                config))
            .collect(toList());
        
        // 3. Distribute tasks to workers
        Map<String, Future<Path>> futures = executorService.submit(tasks);
        
        // 4. Collect results
        Map<String, Path> partitionIndexPaths = new HashMap<>();
        for (Map.Entry<String, Future<Path>> entry : futures.entrySet()) {
            partitionIndexPaths.put(entry.getKey(), entry.getValue().get());
        }
        
        return partitionIndexPaths;
    }
}
```

**Index Building Process**

1. **Data Reading**: Read vectors and row IDs from Iceberg table partition
2. **Vector Normalization**: Apply L2 normalization if using cosine similarity
3. **Index Construction**: Build HNSW graph using JVector
4. **Mapping Creation**: Create node-ID to row-ID mapping
5. **Persistence**: Save index and mapping to S3/HDFS
6. **Metadata Update**: Register index in metadata layer

```java
public class PartitionIndexBuilder {
    public Path buildIndex(
        PartitionInfo partition,
        String columnName,
        VectorIndexConfig config) throws Exception {
        
        // Read vectors from partition
        VectorData vectorData = readVectorsFromPartition(partition, columnName);
        
        // Normalize if needed
        if (config.getSimilarityFunction() == COSINE) {
            normalizeVectors(vectorData.vectors);
        }
        
        // Build index
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(
            vectorData.vectors, 
            vectorData.dimension);
        
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
            ravv, 
            config.getSimilarityFunction());
        
        GraphIndexBuilder builder = new GraphIndexBuilder(
            bsp,
            ravv.dimension(),
            config.getM(),
            config.getEfConstruction(),
            4 * config.getM(),
            1.2f,
            false,
            true);
        
        ImmutableGraphIndex index = builder.build(ravv);
        
        // Create mapping
        NodeRowIdMapping mapping = new NodeRowIdMapping(vectorData.rowIds);
        
        // Save to storage
        Path indexPath = saveIndexToStorage(partition, index, ravv, mapping);
        
        return indexPath;
    }
}
```

#### 3. Search Execution

**Table Function Definition**

```java
@Description("Approximate Nearest Neighbors search for vector similarity")
public class ApproxNearestNeighborsFunction extends AbstractConnectorTableFunction {
    
    private static final String QUERY_VECTOR = "query_vector";
    private static final String COLUMN_NAME = "column_name";
    private static final String LIMIT = "limit";
    
    public ApproxNearestNeighborsFunction() {
        super(
            "system",
            "approx_nearest_neighbors",
            List.of(
                ScalarArgumentSpecification.builder()
                    .name(QUERY_VECTOR)
                    .type(new ArrayType(REAL))
                    .build(),
                ScalarArgumentSpecification.builder()
                    .name(COLUMN_NAME)
                    .type(VARCHAR)
                    .build(),
                ScalarArgumentSpecification.builder()
                    .name(LIMIT)
                    .type(BIGINT)
                    .build()),
            GENERIC_TABLE);
    }
    
    @Override
    public TableFunctionAnalysis analyze(
        ConnectorSession session,
        ConnectorTransactionHandle transaction,
        Map<String, Argument> arguments) {
        
        // Parse arguments
        ScalarArgument queryVector = (ScalarArgument) arguments.get(QUERY_VECTOR);
        ScalarArgument columnName = (ScalarArgument) arguments.get(COLUMN_NAME);
        ScalarArgument limit = (ScalarArgument) arguments.get(LIMIT);
        
        // Parse qualified column name (catalog.schema.table.column)
        QualifiedColumnName qcn = parseQualifiedColumnName(columnName);
        
        // Get index metadata
        VectorIndexMetadata indexMetadata = getIndexMetadata(session, qcn);
        
        // Validate query vector dimension
        validateVectorDimension(queryVector, indexMetadata.getDimension());
        
        // Return analysis with output schema
        Descriptor returnedType = new Descriptor(ImmutableList.of(
            new Descriptor.Field("row_id", Optional.of(BIGINT)),
            new Descriptor.Field("distance", Optional.of(REAL))));
        
        VectorSearchHandle handle = new VectorSearchHandle(
            qcn,
            queryVector,
            limit,
            indexMetadata);
        
        return TableFunctionAnalysis.builder()
            .returnedType(returnedType)
            .handle(handle)
            .build();
    }
}
```

**Distributed Search Execution**

```java
public class DistributedVectorSearchExecutor {
    
    /**
     * Execute search across all partitions
     */
    public SearchResult executeDistributedSearch(
        VectorSearchHandle searchHandle,
        ConnectorSession session) {
        
        // 1. Get relevant partitions (with partition pruning if applicable)
        List<PartitionInfo> partitions = getRelevantPartitions(
            searchHandle.getTableName(),
            session);
        
        // 2. Create search splits for each partition
        List<VectorSearchSplit> splits = partitions.stream()
            .map(partition -> new VectorSearchSplit(
                partition,
                searchHandle.getQueryVector(),
                searchHandle.getLimit(),
                searchHandle.getIndexMetadata()))
            .collect(toList());
        
        // 3. Return splits for distributed execution
        return new SearchResult(splits);
    }
}
```

**Worker-Side Search**

```java
public class VectorSearchPageSource implements ConnectorPageSource {
    
    private final VectorSearchSplit split;
    private boolean finished = false;
    
    @Override
    public Page getNextPage() {
        if (finished) {
            return null;
        }
        
        try {
            // 1. Load index from cache or storage
            OnDiskGraphIndex index = loadIndex(split.getIndexPath());
            NodeRowIdMapping mapping = loadMapping(split.getMappingPath());
            
            // 2. Prepare query vector
            VectorFloat<?> queryVector = prepareQueryVector(
                split.getQueryVector(),
                split.getSimilarityFunction());
            
            // 3. Execute search
            SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(
                queryVector,
                split.getSimilarityFunction(),
                index.getView());
            
            SearchResult result;
            try (GraphSearcher searcher = new GraphSearcher(index)) {
                result = searcher.search(
                    ssp,
                    split.getLimit(),
                    Bits.ALL);
            }
            
            // 4. Build result page with row IDs and distances
            BlockBuilder rowIdBuilder = BIGINT.createBlockBuilder(null, result.size());
            BlockBuilder distanceBuilder = REAL.createBlockBuilder(null, result.size());
            
            for (SearchResult.NodeScore ns : result.getNodes()) {
                long rowId = mapping.getRowId(ns.node);
                BIGINT.writeLong(rowIdBuilder, rowId);
                REAL.writeFloat(distanceBuilder, ns.score);
            }
            
            finished = true;
            return new Page(rowIdBuilder.build(), distanceBuilder.build());
            
        } catch (Exception e) {
            throw new PrestoException(
                VECTOR_SEARCH_ERROR,
                "Error executing vector search", e);
        }
    }
}
```

**Coordinator-Side Aggregation**

```java
public class TopKAggregationOperator implements Operator {
    
    private final PriorityQueue<ScoredRow> topKHeap;
    private final int k;
    
    @Override
    public void addInput(Page page) {
        // Extract row IDs and distances from worker results
        Block rowIdBlock = page.getBlock(0);
        Block distanceBlock = page.getBlock(1);
        
        for (int position = 0; position < page.getPositionCount(); position++) {
            long rowId = BIGINT.getLong(rowIdBlock, position);
            float distance = REAL.getFloat(distanceBlock, position);
            
            // Add to heap, maintaining top-K
            if (topKHeap.size() < k) {
                topKHeap.offer(new ScoredRow(rowId, distance));
            } else if (distance < topKHeap.peek().distance) {
                topKHeap.poll();
                topKHeap.offer(new ScoredRow(rowId, distance));
            }
        }
    }
    
    @Override
    public Page getOutput() {
        // Convert heap to sorted page
        List<ScoredRow> results = new ArrayList<>(topKHeap);
        Collections.sort(results);
        
        BlockBuilder rowIdBuilder = BIGINT.createBlockBuilder(null, results.size());
        BlockBuilder distanceBuilder = REAL.createBlockBuilder(null, results.size());
        
        for (ScoredRow row : results) {
            BIGINT.writeLong(rowIdBuilder, row.rowId);
            REAL.writeFloat(distanceBuilder, row.distance);
        }
        
        return new Page(rowIdBuilder.build(), distanceBuilder.build());
    }
}
```

### SQL API

#### Index Creation

```sql
-- Create a vector index using stored procedure
CALL system.create_vector_index(
    table_name => 'catalog.schema.table_name',
    column_name => 'embedding_column',
    index_name => 'embedding_idx',
    similarity_function => 'COSINE',  -- COSINE, EUCLIDEAN, DOT_PRODUCT
    m => 16,                          -- HNSW M parameter (default: 16)
    ef_construction => 100            -- HNSW ef_construction (default: 100)
);

-- Drop an index
CALL system.drop_vector_index(
    table_name => 'catalog.schema.table_name',
    index_name => 'embedding_idx'
);

-- Refresh an index (rebuild with latest data)
CALL system.refresh_vector_index(
    table_name => 'catalog.schema.table_name',
    index_name => 'embedding_idx'
);

-- List indexes for a table
SELECT * FROM system.vector_indexes 
WHERE table_name = 'catalog.schema.table_name';
```

#### Vector Search

```sql
-- Basic ANN search
SELECT * FROM TABLE(
    approx_nearest_neighbors(
        query_vector => ARRAY[0.1, 0.2, 0.3, ..., 0.768],
        column_name => 'catalog.schema.documents.embedding',
        limit => 10
    )
);

-- Join with original table to get full rows
SELECT d.doc_id, d.title, d.content, ann.distance
FROM documents d
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => ARRAY[0.1, 0.2, 0.3, ..., 0.768],
        column_name => 'catalog.schema.documents.embedding',
        limit => 10
    )
) ann ON d.row_id = ann.row_id
ORDER BY ann.distance;

-- Search with additional filters (partition pruning)
SELECT d.doc_id, d.title, ann.distance
FROM documents d
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => ARRAY[0.1, 0.2, 0.3, ..., 0.768],
        column_name => 'catalog.schema.documents.embedding',
        limit => 10
    )
) ann ON d.row_id = ann.row_id
WHERE d.created_date >= DATE '2024-01-01'
ORDER BY ann.distance;

-- Multiple vector searches in a single query
WITH query_embedding AS (
    SELECT ARRAY[0.1, 0.2, ..., 0.768] AS vec
),
similar_docs AS (
    SELECT d.*, ann.distance
    FROM documents d
    JOIN TABLE(
        approx_nearest_neighbors(
            query_vector => (SELECT vec FROM query_embedding),
            column_name => 'catalog.schema.documents.embedding',
            limit => 100
        )
    ) ann ON d.row_id = ann.row_id
)
SELECT * FROM similar_docs
WHERE distance < 0.5
ORDER BY distance
LIMIT 10;
```

### Metadata Integration

#### Index Metadata Storage

Vector index metadata is stored in the Iceberg metadata layer as custom properties:

```json
{
  "vector_indexes": {
    "embedding_idx": {
      "column_name": "embedding",
      "similarity_function": "COSINE",
      "dimension": 768,
      "m": 16,
      "ef_construction": 100,
      "created_time": 1704470400000,
      "last_modified_time": 1704470400000,
      "status": "READY",
      "total_vectors": 1000000,
      "partitions": {
        "created_date=2024-01-01": "s3://bucket/table/.vector_index/embedding_idx/created_date=2024-01-01/index.hnsw",
        "created_date=2024-01-02": "s3://bucket/table/.vector_index/embedding_idx/created_date=2024-01-02/index.hnsw"
      }
    }
  }
}
```

#### Index Discovery

```java
public class IcebergVectorIndexMetadataStore implements VectorIndexMetadataStore {
    
    @Override
    public Optional<VectorIndexMetadata> getIndexMetadata(
        ConnectorSession session,
        SchemaTableName tableName,
        String columnName) {
        
        // Get Iceberg table
        Table icebergTable = getIcebergTable(session, tableName);
        
        // Read vector index metadata from table properties
        Map<String, String> properties = icebergTable.properties();
        String indexMetadataJson = properties.get("vector_indexes");
        
        if (indexMetadataJson == null) {
            return Optional.empty();
        }
        
        // Parse and find index for column
        Map<String, VectorIndexMetadata> indexes = 
            parseIndexMetadata(indexMetadataJson);
        
        return indexes.values().stream()
            .filter(idx -> idx.getColumnName().equals(columnName))
            .findFirst();
    }
    
    @Override
    public void saveIndexMetadata(
        ConnectorSession session,
        SchemaTableName tableName,
        VectorIndexMetadata metadata) {
        
        // Get Iceberg table
        Table icebergTable = getIcebergTable(session, tableName);
        
        // Update table properties with index metadata
        Map<String, String> updates = new HashMap<>();
        updates.put("vector_indexes", serializeIndexMetadata(metadata));
        
        // Commit metadata update
        icebergTable.updateProperties()
            .set(updates)
            .commit();
    }
}
```

### Partitioning Strategy

#### Partition-Aware Indexing

For partitioned tables, create separate indexes per partition:

```
Table: documents (partitioned by created_date)
├── Partition: created_date=2024-01-01
│   ├── Data files: s3://bucket/table/created_date=2024-01-01/*.parquet
│   └── Index: s3://bucket/table/.vector_index/embedding_idx/created_date=2024-01-01/
│       ├── index.hnsw
│       └── mapping.bin
├── Partition: created_date=2024-01-02
│   ├── Data files: s3://bucket/table/created_date=2024-01-02/*.parquet
│   └── Index: s3://bucket/table/.vector_index/embedding_idx/created_date=2024-01-02/
│       ├── index.hnsw
│       └── mapping.bin
└── ...
```

#### Partition Pruning

Leverage Presto's partition pruning for efficient searches:

```sql
-- Only search partitions matching the filter
SELECT d.doc_id, d.title, ann.distance
FROM documents d
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => ARRAY[...],
        column_name => 'catalog.schema.documents.embedding',
        limit => 10
    )
) ann ON d.row_id = ann.row_id
WHERE d.created_date BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
ORDER BY ann.distance;

-- Presto will only load and search indexes for January 2024 partitions
```

#### Dynamic Partition Management

```java
public class PartitionAwareIndexManager {
    
    /**
     * Build index for new partition
     */
    public void buildPartitionIndex(
        ConnectorSession session,
        SchemaTableName tableName,
        String indexName,
        PartitionInfo newPartition) {
        
        VectorIndexMetadata indexMetadata = 
            getIndexMetadata(session, tableName, indexName);
        
        // Build index for new partition
        Path indexPath = buildIndexForPartition(
            newPartition,
            indexMetadata.getColumnName(),
            indexMetadata.getConfig());
        
        // Update metadata with new partition
        indexMetadata.addPartition(
            newPartition.getPartitionKey(),
            indexPath.toString());
        
        saveIndexMetadata(session, tableName, indexMetadata);
    }
    
    /**
     * Remove index for dropped partition
     */
    public void dropPartitionIndex(
        ConnectorSession session,
        SchemaTableName tableName,
        String indexName,
        PartitionInfo droppedPartition) {
        
        VectorIndexMetadata indexMetadata = 
            getIndexMetadata(session, tableName, indexName);
        
        // Remove partition from metadata
        String indexPath = indexMetadata.removePartition(
            droppedPartition.getPartitionKey());
        
        // Delete index files
        deleteIndexFiles(indexPath);
        
        saveIndexMetadata(session, tableName, indexMetadata);
    }
}
```

### Performance Optimizations

#### 1. Index Caching

Cache frequently accessed indexes on workers:

```java
public class VectorIndexCache {
    
    private final LoadingCache<String, CachedIndex> indexCache;
    
    public VectorIndexCache(long maxCacheSize) {
        this.indexCache = CacheBuilder.newBuilder()
            .maximumSize(maxCacheSize)
            .expireAfterAccess(1, TimeUnit.HOURS)
            .recordStats()
            .build(new CacheLoader<String, CachedIndex>() {
                @Override
                public CachedIndex load(String indexPath) {
                    return loadIndexFromStorage(indexPath);
                }
            });
    }
    
    public OnDiskGraphIndex getIndex(String indexPath) {
        return indexCache.getUnchecked(indexPath).getIndex();
    }
    
    public NodeRowIdMapping getMapping(String indexPath) {
        return indexCache.getUnchecked(indexPath).getMapping();
    }
}
```

#### 2. Adaptive Search Parameters

Dynamically adjust search parameters based on data characteristics:

```java
public class AdaptiveSearchParameterOptimizer {
    
    /**
     * Calculate optimal ef_search based on index size and desired recall
     */
    public int calculateOptimalEfSearch(
        int indexSize,
        int k,
        double targetRecall) {
        
        // Heuristic: ef_search should be at least k
        // and scale with log(indexSize) for better recall
        int baseEf = Math.max(k, 50);
        int scaledEf = (int) (baseEf * Math.log(indexSize) / Math.log(10000));
        
        // Adjust for target recall
        double recallFactor = targetRecall / 0.95;  // 0.95 is baseline
        
        return (int) (scaledEf * recallFactor);
    }
}
```

#### 3. Parallel Index Building

Build multiple partition indexes in parallel:

```java
public class ParallelIndexBuilder {
    
    private final ExecutorService executorService;
    
    public Map<String, Path> buildIndexesInParallel(
        List<PartitionInfo> partitions,
        VectorIndexConfig config) {
        
        // Create tasks
        List<Callable<Pair<String, Path>>> tasks = partitions.stream()
            .map(partition -> (Callable<Pair<String, Path>>) () -> {
                Path indexPath = buildIndexForPartition(partition, config);
                return Pair.of(partition.getPartitionKey(), indexPath);
            })
            .collect(toList());
        
        // Execute in parallel
        try {
            List<Future<Pair<String, Path>>> futures = 
                executorService.invokeAll(tasks);
            
            Map<String, Path> results = new HashMap<>();
            for (Future<Pair<String, Path>> future : futures) {
                Pair<String, Path> result = future.get();
                results.put(result.getLeft(), result.getRight());
            }
            
            return results;
        } catch (Exception e) {
            throw new RuntimeException("Error building indexes in parallel", e);
        }
    }
}
```

#### 4. Memory-Efficient Index Loading

Use memory-mapped files for large indexes:

```java
public class MemoryMappedIndexLoader {
    
    public OnDiskGraphIndex loadIndex(Path indexPath) throws IOException {
        // Use JVector's disk-based index loading
        // which uses memory-mapped files internally
        try (ReaderSupplier rs = ReaderSupplierFactory.open(indexPath)) {
            return OnDiskGraphIndex.load(rs);
        }
    }
}
```

### Configuration

#### Connector Configuration

Add to `catalog/iceberg.properties`:

```properties
# Enable vector similarity search
iceberg.vector-search.enabled=true

# Index cache size (in MB)
iceberg.vector-search.index-cache-size=1024

# Maximum number of parallel index builds
iceberg.vector-search.max-parallel-builds=4

# Default similarity function
iceberg.vector-search.default-similarity-function=COSINE

# Default HNSW parameters
iceberg.vector-search.default-m=16
iceberg.vector-search.default-ef-construction=100
iceberg.vector-search.default-ef-search=50
```

#### Session Properties

```sql
-- Set search parameters for current session
SET SESSION iceberg.vector_search_ef_search = 100;
SET SESSION iceberg.vector_search_timeout = '30s';
```

### Error Handling

```java
public enum VectorSearchErrorCode implements ErrorCodeSupplier {
    VECTOR_INDEX_NOT_FOUND(1, EXTERNAL),
    VECTOR_DIMENSION_MISMATCH(2, USER_ERROR),
    VECTOR_INDEX_CORRUPTED(3, EXTERNAL),
    VECTOR_SEARCH_TIMEOUT(4, EXTERNAL),
    INVALID_SIMILARITY_FUNCTION(5, USER_ERROR);
    
    private final ErrorCode errorCode;
    
    VectorSearchErrorCode(int code, ErrorType type) {
        this.errorCode = new ErrorCode(
            code + 0x0600_0000,  // Vector search error range
            name(),
            type);
    }
    
    @Override
    public ErrorCode toErrorCode() {
        return errorCode;
    }
}
```

## Scalability Considerations

### Handling Large-Scale Data

#### 1. Billion-Scale Datasets

For tables with billions of records:

**Strategy**: Hierarchical partitioning + distributed indexing

```sql
-- Example: 1 billion documents partitioned by date and region
CREATE TABLE documents (
    doc_id BIGINT,
    title VARCHAR,
    content VARCHAR,
    embedding ARRAY(REAL),
    created_date DATE,
    region VARCHAR
) WITH (
    partitioning = ARRAY['created_date', 'region']
);

-- Each partition has ~1M records
-- Total partitions: ~1000 (3 years × 365 days × multiple regions)
-- Each partition index: ~100MB
-- Total index size: ~100GB (distributed across workers)
```

**Benefits**:
- Parallel index building across 1000+ partitions
- Efficient partition pruning reduces search space
- Worker-local index caching
- Horizontal scalability

#### 2. Memory Management

```java
public class VectorSearchMemoryManager {
    
    /**
     * Estimate memory requirements for index
     */
    public long estimateIndexMemory(
        int numVectors,
        int dimension,
        int m) {
        
        // HNSW memory estimation
        long graphMemory = numVectors * m * 2 * 4;  // edges (int)
        long vectorMemory = numVectors * dimension * 4;  // vectors (float)
        long mappingMemory = numVectors * 8;  // row IDs (long)
        long overhead = (long) ((graphMemory + vectorMemory) * 0.1);
        
        return graphMemory + vectorMemory + mappingMemory + overhead;
    }
    
    /**
     * Check if index can be loaded in available memory
     */
    public boolean canLoadIndex(String indexPath) {
        long indexSize = getIndexSize(indexPath);
        long availableMemory = getAvailableMemory();
        
        return indexSize < availableMemory * 0.8;  // 80% threshold
    }
}
```

#### 3. Incremental Index Updates

Support for incremental updates without full rebuild:

```java
public class IncrementalIndexUpdater {
    
    /**
     * Add new vectors to existing index
     */
    public void addVectors(
        Path indexPath,
        List<float[]> newVectors,
        List<Long> newRowIds) {
        
        // Load existing index
        OnDiskGraphIndex existingIndex = loadIndex(indexPath);
        NodeRowIdMapping existingMapping = loadMapping(indexPath);
        
        // Create incremental index for new vectors
        OnDiskGraphIndex incrementalIndex = buildIncrementalIndex(
            newVectors,
            existingIndex);
        
        // Merge mappings
        NodeRowIdMapping mergedMapping = existingMapping.merge(newRowIds);
        
        // Save merged index
        saveIndex(indexPath, incrementalIndex, mergedMapping);
    }
}
```

### Query Performance

#### Expected Performance Characteristics

| Dataset Size | Partitions | Index Size/Partition | Search Latency (p50) | Search Latency (p99) |
|--------------|------------|---------------------|---------------------|---------------------|
| 1M vectors   | 1          | 100 MB              | 10 ms               | 50 ms               |
| 10M vectors  | 10         | 100 MB              | 15 ms               | 75 ms               |
| 100M vectors | 100        | 100 MB              | 20 ms               | 100 ms              |
| 1B vectors   | 1000       | 100 MB              | 30 ms               | 150 ms              |

**Assumptions**:
- 768-dimensional vectors
- HNSW parameters: M=16, ef_construction=100, ef_search=50
- Top-10 search
- Warm cache
- 10 worker nodes

#### Optimization Techniques

1. **Partition Pruning**: Reduce search space by 10-100x
2. **Index Caching**: Eliminate S3 I/O for hot indexes
3. **Parallel Search**: Linear scalability with worker count
4. **Adaptive ef_search**: Balance accuracy vs. speed
5. **Result Streaming**: Start returning results before all partitions complete

## Testing Strategy

### Unit Tests

```java
@Test
public void testVectorIndexBuilding() {
    // Test index creation with various parameters
    VectorIndexBuilder builder = new VectorIndexBuilder();
    
    List<float[]> vectors = generateTestVectors(1000, 128);
    List<Long> rowIds = generateRowIds(1000);
    
    Path indexPath = builder.buildIndex(
        vectors,
        rowIds,
        VectorSimilarityFunction.COSINE,
        16,
        100);
    
    assertTrue(Files.exists(indexPath));
    
    // Verify index can be loaded
    OnDiskGraphIndex index = loadIndex(indexPath);
    assertEquals(1000, index.size());
}

@Test
public void testVectorSearch() {
    // Test search accuracy
    OnDiskGraphIndex index = loadTestIndex();
    float[] queryVector = generateTestVector(128);
    
    SearchResult result = searchIndex(index, queryVector, 10);
    
    assertEquals(10, result.size());
    assertTrue(result.getNodes().get(0).score >= 
               result.getNodes().get(9).score);
}

@Test
public void testTopKAggregation() {
    // Test coordinator aggregation
    TopKAggregator aggregator = new TopKAggregator(10);
    
    // Simulate results from 3 workers
    aggregator.addResults(generateWorkerResults(10, 0.1f, 0.5f));
    aggregator.addResults(generateWorkerResults(10, 0.2f, 0.6f));
    aggregator.addResults(generateWorkerResults(10, 0.15f, 0.55f));
    
    List<ScoredRow> topK = aggregator.getTopK();
    
    assertEquals(10, topK.size());
    // Verify global top-K is correct
    assertTrue(topK.get(0).distance <= 0.1f);
}
```

### Integration Tests

```java
@Test
public void testEndToEndVectorSearch() {
    // Create test table with embeddings
    createTestTable("test_vectors", 10000, 768);
    
    // Create index
    executeQuery("CALL system.create_vector_index(" +
        "table_name => 'test.test_vectors', " +
        "column_name => 'embedding', " +
        "index_name => 'emb_idx')");
    
    // Execute search
    float[] queryVector = generateTestVector(768);
    String query = String.format(
        "SELECT * FROM TABLE(approx_nearest_neighbors(" +
        "query_vector => ARRAY[%s], " +
        "column_name => 'test.test_vectors.embedding', " +
        "limit => 10))",
        arrayToString(queryVector));
    
    MaterializedResult result = executeQuery(query);
    
    assertEquals(10, result.getRowCount());
}

@Test
public void testPartitionedVectorSearch() {
    // Create partitioned table
    createPartitionedTable("test_vectors_partitioned", 100000, 768, "created_date");
    
    // Create index
    executeQuery("CALL system.create_vector_index(" +
        "table_name => 'test.test_vectors_partitioned', " +
        "column_name => 'embedding', " +
        "index_name => 'emb_idx')");
    
    // Search with partition filter
    String query = 
        "SELECT v.*, ann.distance " +
        "FROM test_vectors_partitioned v " +
        "JOIN TABLE(approx_nearest_neighbors(" +
        "  query_vector => ARRAY[...], " +
        "  column_name => 'test.test_vectors_partitioned.embedding', " +
        "  limit => 10)) ann ON v.row_id = ann.row_id " +
        "WHERE v.created_date = DATE '2024-01-01'";
    
    MaterializedResult result = executeQuery(query);
    
    // Verify only one partition was searched
    assertEquals(10, result.getRowCount());
}
```

### Performance Tests

```java
@Test
public void testSearchLatency() {
    // Benchmark search performance
    OnDiskGraphIndex index = loadLargeTestIndex(1_000_000, 768);
    
    List<Long> latencies = new ArrayList<>();
    
    for (int i = 0; i < 1000; i++) {
        float[] queryVector = generateRandomVector(768);
        
        long start = System.nanoTime();
        SearchResult result = searchIndex(index, queryVector, 10);
        long end = System.nanoTime();
        
        latencies.add((end - start) / 1_000_000);  // Convert to ms
    }
    
    // Calculate percentiles
    Collections.sort(latencies);
    long p50 = latencies.get(500);
    long p95 = latencies.get(950);
    long p99 = latencies.get(990);
    
    // Assert performance targets
    assertTrue(p50 < 20, "P50 latency should be < 20ms");
    assertTrue(p95 < 50, "P95 latency should be < 50ms");
    assertTrue(p99 < 100, "P99 latency should be < 100ms");
}

@Test
public void testScalability() {
    // Test scalability with increasing data size
    int[] sizes = {10_000, 100_000, 1_000_000, 10_000_000};
    
    for (int size : sizes) {
        OnDiskGraphIndex index = buildTestIndex(size, 768);
        
        long avgLatency = measureAverageSearchLatency(index, 100);
        
        // Latency should scale sub-linearly
        assertTrue(avgLatency < size / 10000 * 20);
    }
}
```

### Accuracy Tests

```java
@Test
public void testSearchAccuracy() {
    // Test recall@k
    List<float[]> vectors = generateTestVectors(10000, 128);
    OnDiskGraphIndex index = buildIndex(vectors);
    
    // For each vector, search and verify it finds itself
    int correctResults = 0;
    
    for (int i = 0; i < 100; i++) {
        float[] queryVector = vectors.get(i);
        SearchResult result = searchIndex(index, queryVector, 10);
        
        // Check if the query vector itself is in top-10
        if (result.getNodes().stream()
            .anyMatch(ns -> ns.node == i)) {
            correctResults++;
        }
    }
    
    double recall = correctResults / 100.0;
    assertTrue(recall > 0.95, "Recall should be > 95%");
}
```

## Adoption Plan

### Phase 1: Core Infrastructure (Months 1-2)

**Goals**:
- Implement basic vector index metadata layer
- Integrate JVector library
- Create index building infrastructure
- Implement single-partition index support

**Deliverables**:
- `VectorIndexMetadata` and `VectorIndexManager` interfaces
- `VectorIndexBuilder` for single partition
- Metadata storage in Iceberg properties
- Basic stored procedures for index management

**Success Criteria**:
- Can create and query indexes on non-partitioned tables
- Index metadata persisted and discoverable
- Basic unit tests passing

### Phase 2: Table Function Integration (Months 2-3)

**Goals**:
- Implement `approx_nearest_neighbors` table function
- Create search execution infrastructure
- Implement basic result aggregation

**Deliverables**:
- `ApproxNearestNeighborsFunction` implementation
- `VectorSearchPageSource` for search execution
- `TopKAggregator` for result merging
- Integration with Presto's TVF framework

**Success Criteria**:
- Can execute vector searches via SQL
- Results are accurate (>95% recall)
- Integration tests passing

### Phase 3: Distributed Execution (Months 3-4)

**Goals**:
- Implement partitioned index support
- Enable distributed index building
- Implement parallel search execution
- Add partition pruning

**Deliverables**:
- `DistributedVectorIndexBuilder`
- Partition-aware index management
- Parallel search across workers
- Partition pruning optimization

**Success Criteria**:
- Can handle partitioned tables with 100+ partitions
- Index building parallelized across workers
- Search executes in parallel
- Partition pruning reduces search time

### Phase 4: Optimizations (Months 4-5)

**Goals**:
- Implement index caching
- Add adaptive search parameters
- Optimize memory usage
- Performance tuning

**Deliverables**:
- `VectorIndexCache` implementation
- Adaptive ef_search calculation
- Memory-efficient index loading
- Performance benchmarks

**Success Criteria**:
- P99 latency < 100ms for 1M vectors
- Can handle 1B+ vectors with partitioning
- Memory usage optimized
- Cache hit rate > 80%

### Phase 5: Advanced Features (Months 5-6)

**Goals**:
- Incremental index updates
- Multiple similarity functions
- Advanced query patterns
- Production hardening

**Deliverables**:
- Incremental index update support
- Support for all similarity functions
- Comprehensive error handling
- Production monitoring and metrics

**Success Criteria**:
- Can update indexes without full rebuild
- All similarity functions working
- Production-ready error handling
- Monitoring dashboards available

### Migration Path

For users adopting this feature:

1. **Table Preparation**:
   ```sql
   -- Ensure table has row_id column
   ALTER TABLE documents ADD COLUMN row_id BIGINT;
   
   -- Populate row_id (one-time operation)
   UPDATE documents SET row_id = ROW_NUMBER() OVER ();
   ```

2. **Index Creation**:
   ```sql
   -- Create initial index
   CALL system.create_vector_index(
       table_name => 'catalog.schema.documents',
       column_name => 'embedding',
       index_name => 'embedding_idx'
   );
   ```

3. **Query Migration**:
   ```sql
   -- Before: Brute force (slow)
   SELECT doc_id, title,
          array_sum(transform(
              zip_with(embedding, query_vec, (x, y) -> x * y),
              x -> x
          )) AS similarity
   FROM documents
   ORDER BY similarity DESC
   LIMIT 10;
   
   -- After: ANN search (fast)
   SELECT d.doc_id, d.title, ann.distance
   FROM documents d
   JOIN TABLE(approx_nearest_neighbors(
       query_vector => query_vec,
       column_name => 'catalog.schema.documents.embedding',
       limit => 10
   )) ann ON d.row_id = ann.row_id
   ORDER BY ann.distance;
   ```

### Backward Compatibility

- No breaking changes to existing Presto APIs
- New feature is opt-in via configuration
- Existing queries continue to work unchanged
- Index creation is explicit (not automatic)

## Alternative Approaches Considered

### 1. External Vector Database Integration

**Approach**: Integrate with external vector databases (Pinecot, Milvus, Weaviate)

**Pros**:
- Mature, specialized solutions
- Advanced features (filtering, hybrid search)
- Proven scalability

**Cons**:
- Data movement overhead
- Additional infrastructure complexity
- Increased latency
- Data consistency challenges
- Additional licensing/costs

**Decision**: Rejected in favor of native integration for better performance and simpler architecture

### 2. Brute Force with GPU Acceleration

**Approach**: Use GPU-accelerated brute force search

**Pros**:
- 100% recall
- Simpler implementation
- No index maintenance

**Cons**:
- Requires GPU infrastructure
- Doesn't scale to billions of vectors
- Higher cost per query
- Limited GPU availability in cloud

**Decision**: Rejected due to scalability limitations

### 3. LSH (Locality Sensitive Hashing)

**Approach**: Use LSH-based approximate search

**Pros**:
- Simpler than HNSW
- Good for very high dimensions
- Easier to distribute

**Cons**:
- Lower recall than HNSW
- Requires more memory
- Less mature Java implementations

**Decision**: Rejected in favor of HNSW's better accuracy/performance tradeoff

### 4. Faiss Integration via JNI

**Approach**: Integrate Facebook's Faiss library via JNI

**Pros**:
- Industry-standard library
- Excellent performance
- Many index types

**Cons**:
- JNI complexity and overhead
- Platform-specific binaries
- Harder to debug and maintain
- Licensing considerations

**Decision**: Rejected in favor of pure Java solution (JVector)

## Security Considerations

### 1. Access Control

Vector indexes respect existing Presto access controls:

```java
public class VectorIndexAccessControl {
    
    public void checkCanCreateIndex(
        ConnectorSession session,
        SchemaTableName tableName) {
        
        // Require INSERT privilege on table
        accessControl.checkCanInsertIntoTable(
            session.toSecurityContext(),
            tableName);
    }
    
    public void checkCanSearchIndex(
        ConnectorSession session,
        SchemaTableName tableName) {
        
        // Require SELECT privilege on table
        accessControl.checkCanSelectFromTable(
            session.toSecurityContext(),
            tableName);
    }
}
```

### 2. Data Privacy

- Indexes stored in same location as table data
- Inherit table's encryption settings
- Support for encrypted S3 buckets
- No data leakage through index metadata

### 3. Resource Limits

```java
public class VectorSearchResourceLimits {
    
    // Prevent resource exhaustion
    private static final int MAX_QUERY_VECTOR_DIMENSION = 4096;
    private static final int MAX_SEARCH_LIMIT = 10000;
    private static final long MAX_INDEX_SIZE_BYTES = 10L * 1024 * 1024 * 1024;  // 10GB
    
    public void validateSearchRequest(VectorSearchRequest request) {
        if (request.getQueryVector().length > MAX_QUERY_VECTOR_DIMENSION) {
            throw new PrestoException(
                INVALID_ARGUMENTS,
                "Query vector dimension exceeds maximum: " + MAX_QUERY_VECTOR_DIMENSION);
        }
        
        if (request.getLimit() > MAX_SEARCH_LIMIT) {
            throw new PrestoException(
                INVALID_ARGUMENTS,
                "Search limit exceeds maximum: " + MAX_SEARCH_LIMIT);
        }
    }
}
```

## Monitoring and Observability

### Metrics

```java
public class VectorSearchMetrics {
    
    // Index building metrics
    private final Counter indexBuildsStarted;
    private final Counter indexBuildsCompleted;
    private final Counter indexBuildsFailed;
    private final Timer indexBuildDuration;
    
    // Search metrics
    private final Counter searchesExecuted;
    private final Timer searchLatency;
    private final Histogram searchResultSize;
    private final Counter cacheHits;
    private final Counter cacheMisses;
    
    // Resource metrics
    private final Gauge indexCacheSize;
    private final Gauge indexCacheMemoryUsage;
    private final Counter indexLoads;
    private final Timer indexLoadDuration;
}
```

### Logging

```java
// Index building
log.info("Building vector index for table %s, column %s, partition %s",
    tableName, columnName, partition);
log.info("Index built successfully: %d vectors, dimension %d, took %dms",
    vectorCount, dimension, duration);

// Search execution
log.debug("Executing vector search: table=%s, limit=%d, partitions=%d",
    tableName, limit, partitionCount);
log.debug("Search completed: found %d results in %dms",
    resultCount, duration);

// Errors
log.error("Failed to build index for partition %s: %s",
    partition, error.getMessage(), error);
```

### Tracing

Integration with Presto's distributed tracing:

```java
@Traced
public SearchResult executeVectorSearch(VectorSearchRequest request) {
    Span span = tracer.getCurrentSpan();
    span.setAttribute("table", request.getTableName());
    span.setAttribute("limit", request.getLimit());
    span.setAttribute("partitions", request.getPartitionCount());
    
    try {
        SearchResult result = doSearch(request);
        span.setAttribute("results", result.size());
        return result;
    } catch (Exception e) {
        span.recordException(e);
        throw e;
    }
}
```

## Documentation Plan

### User Documentation

1. **Getting Started Guide**
   - Prerequisites (table requirements)
   - Creating first index
   - Running first search
   - Common patterns

2. **SQL Reference**
   - `approx_nearest_neighbors` function
   - Stored procedures
   - System tables
   - Configuration properties

3. **Best Practices**
   - Choosing similarity functions
   - Tuning HNSW parameters
   - Partitioning strategies
   - Performance optimization

4. **Troubleshooting**
   - Common errors
   - Performance issues
   - Index corruption recovery

### Developer Documentation

1. **Architecture Overview**
   - Component diagram
   - Data flow
   - Integration points

2. **API Reference**
   - SPI interfaces
   - Extension points
   - Custom implementations

3. **Contributing Guide**
   - Development setup
   - Testing guidelines
   - Code style

## Future Enhancements

### Short Term (6-12 months)

1. **Filtered Search**
   ```sql
   -- Search with metadata filters
   SELECT * FROM TABLE(
       approx_nearest_neighbors(
           query_vector => ARRAY[...],
           column_name => 'documents.embedding',
           limit => 10,
           filter => 'category = ''technology'' AND date > DATE ''2024-01-01'''
       )
   );
   ```

2. **Hybrid Search**
   - Combine vector similarity with keyword search
   - Weighted scoring

3. **Multiple Vector Columns**
   - Support multiple embeddings per row
   - Multi-modal search

### Medium Term (12-24 months)

1. **Quantization**
   - Product quantization for memory efficiency
   - Scalar quantization for faster search

2. **GPU Acceleration**
   - Optional GPU support for index building
   - GPU-accelerated search

3. **Advanced Index Types**
   - IVF (Inverted File Index)
   - DiskANN
   - Hierarchical indexes

### Long Term (24+ months)

1. **Automatic Index Management**
   - Auto-create indexes based on query patterns
   - Auto-tune parameters
   - Auto-refresh on data changes

2. **Federated Vector Search**
   - Search across multiple catalogs
   - Cross-connector search

3. **Vector Analytics**
   - Clustering
   - Dimensionality reduction
   - Anomaly detection

## Conclusion

This RFC proposes a comprehensive solution for vector similarity search in Presto using JVector. The design leverages Presto's distributed architecture for scalable, efficient ANN search while maintaining simplicity and ease of use.

Key benefits:
- **Native Integration**: No external dependencies or data movement
- **Scalability**: Handles billions of vectors through partitioning
- **Performance**: Sub-100ms latency for most queries
- **Ease of Use**: Simple SQL interface via table functions
- **Flexibility**: Supports multiple similarity functions and use cases

The phased adoption plan ensures a smooth rollout with minimal risk, while the extensible architecture allows for future enhancements.

## References

1. JVector Library: https://github.com/jbellis/jvector
2. HNSW Algorithm: Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
3. Presto Table Functions RFC-0020: [Internal reference]
4. Iceberg Table Format: https://iceberg.apache.org/
5. Vector Database Benchmarks: https://github.com/erikbern/ann-benchmarks

## Appendix A: Example Use Cases

### Use Case 1: Semantic Document Search

```sql
-- Create documents table with embeddings
CREATE TABLE knowledge_base (
    doc_id BIGINT,
    row_id BIGINT,
    title VARCHAR,
    content VARCHAR,
    embedding ARRAY(REAL),  -- 768-dim from BERT
    category VARCHAR,
    created_date DATE
) WITH (
    partitioning = ARRAY['created_date']
);

-- Create vector index
CALL system.create_vector_index(
    table_name => 'knowledge_base',
    column_name => 'embedding',
    index_name => 'kb_embedding_idx',
    similarity_function => 'COSINE'
);

-- Search for similar documents
WITH query_embedding AS (
    SELECT get_embedding('How to optimize database queries?') AS vec
)
SELECT 
    kb.title,
    kb.content,
    kb.category,
    ann.distance AS similarity_score
FROM knowledge_base kb
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => (SELECT vec FROM query_embedding),
        column_name => 'knowledge_base.embedding',
        limit => 20
    )
) ann ON kb.row_id = ann.row_id
WHERE kb.created_date >= DATE '2024-01-01'
ORDER BY ann.distance
LIMIT 10;
```

### Use Case 2: Product Recommendations

```sql
-- Products with image embeddings
CREATE TABLE products (
    product_id BIGINT,
    row_id BIGINT,
    name VARCHAR,
    description VARCHAR,
    image_embedding ARRAY(REAL),  -- 512-dim from ResNet
    category VARCHAR,
    price DECIMAL(10,2)
);

-- Find visually similar products
SELECT 
    p.product_id,
    p.name,
    p.price,
    ann.distance AS visual_similarity
FROM products p
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => (SELECT image_embedding FROM products WHERE product_id = 12345),
        column_name => 'products.image_embedding',
        limit => 50
    )
) ann ON p.row_id = ann.row_id
WHERE p.category = 'electronics'
  AND p.price BETWEEN 100 AND 500
ORDER BY ann.distance
LIMIT 10;
```

### Use Case 3: Anomaly Detection

```sql
-- User behavior embeddings
CREATE TABLE user_sessions (
    session_id BIGINT,
    row_id BIGINT,
    user_id BIGINT,
    behavior_embedding ARRAY(REAL),  -- 128-dim behavioral features
    timestamp TIMESTAMP,
    is_fraud BOOLEAN
);

-- Find anomalous sessions (far from normal patterns)
WITH normal_pattern AS (
    SELECT AVG_EMBEDDING(behavior_embedding) AS avg_vec
    FROM user_sessions
    WHERE is_fraud = false
      AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
)
SELECT 
    s.session_id,
    s.user_id,
    ann.distance AS anomaly_score
FROM user_sessions s
JOIN TABLE(
    approx_nearest_neighbors(
        query_vector => (SELECT avg_vec FROM normal_pattern),
        column_name => 'user_sessions.behavior_embedding',
        limit => 1000
    )
) ann ON s.row_id = ann.row_id
WHERE ann.distance > 0.8  -- High distance = anomalous
ORDER BY ann.distance DESC
LIMIT 100;
```

## Appendix B: Performance Tuning Guide

### HNSW Parameter Selection

| Use Case | M | ef_construction | ef_search | Trade-off |
|----------|---|-----------------|-----------|-----------|
| High Recall | 32 | 200 | 100 | Slower build, larger index, better accuracy |
| Balanced | 16 | 100 | 50 | Good balance (recommended) |
| Fast Search | 8 | 50 | 25 | Faster search, lower accuracy |
| Memory Constrained | 8 | 50 | 25 | Smaller index size |

### Partition Size Guidelines

| Total Vectors | Recommended Partition Size | Number of Partitions | Rationale |
|---------------|---------------------------|---------------------|-----------|
| < 1M | No partitioning | 1 | Single index sufficient |
| 1M - 10M | 1M per partition | 1-10 | Balance between parallelism and overhead |
| 10M - 100M | 1M per partition | 10-100 | Good parallelism |
| 100M - 1B | 1M per partition | 100-1000 | Maximum parallelism |
| > 1B | 1M per partition | 1000+ | Hierarchical partitioning recommended |

### Cache Configuration

```properties
# For 100GB total index size across all partitions
# With 10 worker nodes, each worker handles ~10GB

# Per-worker cache size (cache hot indexes)
iceberg.vector-search.index-cache-size=2048  # 2GB per worker

# This allows caching ~20% of indexes per worker
# Adjust based on query patterns and available memory