public AbstractBounds<T> deserialize(DataInput in, IPartitioner p, int version) throws IOException {
    boolean isToken, startInclusive, endInclusive;
    if (version < MessagingService.VERSION_30) {
        int kind = in.readInt();
        isToken = kind >= 0;
        if (!isToken)
            kind = -(kind + 1);
        // Pre-3.0, everything that wasa not a Range was (wrongly) serialized as a Bound;
        startInclusive = kind != Type.RANGE.ordinal();
        endInclusive = true;
    } else {
        int flags = in.readUnsignedByte();
        isToken = (flags & IS_TOKEN_FLAG) != 0;
        startInclusive = (flags & START_INCLUSIVE_FLAG) != 0;
        endInclusive = (flags & END_INCLUSIVE_FLAG) != 0;
    }
    T left = serializer.deserialize(in, p, version);
    T right = serializer.deserialize(in, p, version);
    assert isToken == left instanceof Token;
    if (startInclusive)
        return endInclusive ? new Bounds<T>(left, right) : new IncludingExcludingBounds<T>(left, right);
    else
        return endInclusive ? new Range<T>(left, right) : new ExcludingBounds<T>(left, right);
}
