public boolean isEmpty() {
    Holder holder = holder();
    return holder.deletionInfo.isLive() && BTree.isEmpty(holder.tree) && holder.staticRow.isEmpty();
}
