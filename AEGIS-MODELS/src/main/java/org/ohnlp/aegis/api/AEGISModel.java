package org.ohnlp.aegis.api;

import com.fasterxml.jackson.databind.JsonNode;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;

public interface AEGISModel {

    /**
     * Initializes and fits a new AEGIS model and associated normalizers using the supplied unnormalized dataset and model configuration
     *
     * @param dataSet     The dataset to fit on
     * @param modelConfig A model configuration json, model defined
     * @return The trained model.
     * @throws IOException If an error occurs during model initialization
     */
    AEGISModel initializeAndFit(File modelDirectory, DataSet dataSet, JsonNode modelConfig) throws IOException;

    /**
     * Scores a specific data element on the given model
     *
     * @param data The data to score, unnormalized
     * @return The model score
     */
    double score(DataSet data);

    /**
     * Loads a model from a directory
     *
     * @param modelDirectory The directory to load the model from, saved via #serialize
     * @return The model with weights loaded from the specified directory
     * @throws IOException If an error occurs during deserialization
     */
    AEGISModel deserialize(File modelDirectory) throws IOException;

    /**
     * Saves the model to the specified directory
     * @param modelDirectory The directory to serialize to
     * @throws IOException If an error occurs during serialization
     */
    void serialize(File modelDirectory) throws IOException;
}
