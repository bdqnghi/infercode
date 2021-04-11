/**
 * return true if @param range intersects any of the given @param ranges
 */
public boolean intersects(Iterable<Range<T>> ranges) {
    for (Range<T> range2 : ranges) {
        if (range2.intersects(this))
            return true;
    }
    return false;
}
