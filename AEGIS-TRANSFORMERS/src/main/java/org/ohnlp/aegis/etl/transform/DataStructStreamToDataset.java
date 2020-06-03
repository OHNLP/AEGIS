package org.ohnlp.aegis.etl.transform;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.ohnlp.aegis.etl.DataStruct;

import java.util.stream.Stream;

/**
 * Converts DataStruct streams produced by the extractor/filters to a ND4J {@link DataSet}
 */
public class DataStructStreamToDataset {
    /**
     * Creates a {@link DataSet} usable to train or evaluate models from the input stream
     * @param stream The stream to transform
     * @return A Corresponding ND4J {@link DataSet}
     */
    public DataSet fromStream(Stream<DataStruct> stream) {
        double[][] data = stream.map(DataStruct::getData).toArray(double[][]::new);
        return new DataSet(Nd4j.create(data), Nd4j.create(data));
        // TODO decouple this from autoencoder implementation when other models are implemented. Labels not always equal to input
    }
}
