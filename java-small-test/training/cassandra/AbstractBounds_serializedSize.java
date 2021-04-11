public long serializedSize(AbstractBounds<T> ab, int version) {
    int size = version < MessagingService.VERSION_30 ? TypeSizes.sizeof(kindInt(ab)) : 1;
    size += serializer.serializedSize(ab.left, version);
    size += serializer.serializedSize(ab.right, version);
    return size;
}
