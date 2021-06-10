public static <T extends RingPosition<T>> AbstractBounds<T> bounds(T min, boolean inclusiveMin, T max, boolean inclusiveMax) {
    if (inclusiveMin && inclusiveMax)
        return new Bounds<T>(min, max);
    else if (inclusiveMax)
        return new Range<T>(min, max);
    else if (inclusiveMin)
        return new IncludingExcludingBounds<T>(min, max);
    else
        return new ExcludingBounds<T>(min, max);
}
