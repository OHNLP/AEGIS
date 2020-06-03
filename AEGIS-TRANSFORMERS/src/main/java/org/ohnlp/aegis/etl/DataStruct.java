package org.ohnlp.aegis.etl;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Date;

/**
 * Represents a distinct data point at a certain date
 */
public class DataStruct implements Comparable<DataStruct> {
    private Date date;
    private double[] data;

    public DataStruct(Date date, double[] data) {
        this.date = date;
        this.data = data;
    }

    public Date getDate() {
        return date;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public double[] getData() {
        return data;
    }

    public void setData(double[] data) {
        this.data = data;
    }

    public int compareTo(DataStruct o) {
        int ret = this.date.compareTo(o.date);
        if (ret == 0) {
            return Integer.compare(this.hashCode(), o.hashCode()); // TODO should we even allow two points on the same date realistically?
        } else {
            return ret;
        }
    }
}
