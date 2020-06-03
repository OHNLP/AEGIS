package org.ohnlp.aegis.etl.filters;

import org.ohnlp.aegis.etl.DataStruct;

import java.util.Date;
import java.util.List;
import java.util.stream.Stream;

/**
 * Filters  dates within certain set of date/time range
 */
public class DateRangeFilter {

    private final Stream<DataStruct> input;
    private final List<Date[]> dateRanges;

    public DateRangeFilter(Stream<DataStruct> input, List<Date[]> dateRanges) {
        this.input = input;
        this.dateRanges = dateRanges;
    }

    public Stream<DataStruct> getInDateRange() {
        return this.input.filter(d -> {
            for (Date[] range : dateRanges) {
                if (range[0].getTime() <= d.getDate().getTime() && d.getDate().getTime() < range[1].getTime()) {
                    return true;
                }
            }
            return false;
        });
    }

    public Stream<DataStruct> getOutsideDateRange() {
        return this.input.filter(d -> {
            for (Date[] range : dateRanges) {
                if (range[0].getTime() <= d.getDate().getTime() && d.getDate().getTime() < range[1].getTime()) {
                    return false;
                }
            }
            return true;
        });
    }
}
