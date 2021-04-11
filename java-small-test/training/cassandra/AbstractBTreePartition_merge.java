private RowAndDeletionMergeIterator merge(Iterator<Row> rowIter, Iterator<RangeTombstone> deleteIter, ColumnFilter selection, boolean reversed, Holder current, Row staticRow) {
    return new RowAndDeletionMergeIterator(metadata, partitionKey, current.deletionInfo.getPartitionDeletion(), selection, staticRow, reversed, current.stats, rowIter, deleteIter, canHaveShadowedData());
}
