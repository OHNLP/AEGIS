package org.ohnlp.aegis.etl.filters;

import de.jollyday.HolidayCalendar;
import de.jollyday.HolidayManager;
import de.jollyday.ManagerParameters;
import org.ohnlp.aegis.etl.DataStruct;

import java.util.Calendar;
import java.util.List;
import java.util.Properties;
import java.util.stream.Stream;

/**
 * Filters weekends and holidays from a given dataset.
 * This is desirable because the patient characteristics seen differ wildly for these dates from typical weekdays
 */
public class WeekendAndHolidayFilter {
    private final Stream<DataStruct> input;
    private final HolidayManager holidayManager;

    public WeekendAndHolidayFilter(Stream<DataStruct> input, HolidayCalendar country) {
        this.input = input;
        this.holidayManager = HolidayManager.getInstance(ManagerParameters.create(country, (Properties) null));
        // TODO federal holidays only
    }

    public Stream<DataStruct> getNormalDays() {
        Calendar c = Calendar.getInstance();
        return this.input.filter(d -> {
            c.setTime(d.getDate());
            return !(
                    c.get(Calendar.DAY_OF_WEEK) == Calendar.SATURDAY
                            || c.get(Calendar.DAY_OF_WEEK) == Calendar.SUNDAY
            ) && !holidayManager.isHoliday(c);
        });
    }

    public Stream<DataStruct> getWeekendsAndHolidays() {
        Calendar c = Calendar.getInstance();
        return this.input.filter(d -> {
            c.setTime(d.getDate());
            return c.get(Calendar.DAY_OF_WEEK) == Calendar.SATURDAY
                    || c.get(Calendar.DAY_OF_WEEK) == Calendar.SUNDAY
                    || holidayManager.isHoliday(c);
        });
    }
}
