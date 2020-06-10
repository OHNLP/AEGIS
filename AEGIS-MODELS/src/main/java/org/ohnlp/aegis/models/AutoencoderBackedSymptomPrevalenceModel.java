package org.ohnlp.aegis.models;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.ohnlp.aegis.api.AEGISModel;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * A stacked autoencoder model using symptom-prevalence distributions to perform aberration detection
 */
public class AutoencoderBackedSymptomPrevalenceModel implements AEGISModel {
    private HyperParameters parameters;
    private ModelScore scores;
    private NormalizerMinMaxScaler normalizer;
    private MultiLayerNetwork model;

    /**
     * Creates a new stacked autoencoder model instance trained on the supplied dataset with the listed layer dimensions
     * @param modelDirectory A directory to which to write the trained model
     * @param data The data to fit the model on
     * @param dims The sequential dimensions for the encoder portion of the stacked autoencoder. This will be reversed
     *             automatically for the decoder portion
     * @return A trained model instance that is also saved to the supplied model directory
     * @throws IOException if an error occurs during the serialization process
     */
    public AutoencoderBackedSymptomPrevalenceModel initModelFittedOnData(File modelDirectory, DataSet data, int... dims) throws IOException {
        this.initializeAndFit(data, dims);
        this.serialize(modelDirectory);
        return this;
    }

    /**
     * Loads a previously saved model (via {@link #serialize(File)})
     * @param modelDirectory The directory in which the model is saved
     * @return The loaded stacked autoencoder model
     * @throws IOException If an error occurs during model load
     */
    public AutoencoderBackedSymptomPrevalenceModel deserialize(File modelDirectory) throws IOException {
        load(modelDirectory);
        return this;
    }

    /**
     * Scores a specific data element
     * @param data The data to score, unnormalized
     * @return The model score
     */
    @Override
    public double score(DataSet data) {
        this.normalizer.transform(data);
        return this.model.score(data);
    }

    /**
     * @return The local aberration threshold for this trained model, defined as mean + 2 standard deviations of error over test data
     */
    public double localAberrationThreshold() {
        return this.scores.getMean() + (this.scores.stdDev * 2);
    }


    @Override
    public AEGISModel initializeAndFit(File modelDirectory, DataSet dataSet, JsonNode modelConfig) throws IOException {
        if (!modelConfig.has("dims")) {
            throw new IllegalArgumentException("A dimension parameter of encoder dimension sizes is required");
        }
        List<Integer> dims = new LinkedList<>();
        for (JsonNode dim : modelConfig.get("dims")) {
            try {
                dims.add(dim.asInt());
            } catch (Throwable t) {
                throw new IOException(t);
            }
        }
        return initializeAndFit(dataSet, dims.stream().mapToInt(i -> i).toArray());
    }


    /**
     * Fits the model on an initial dataset
     * @param dataset The dataset
     * @param dims The encoder layer sizes
     * @return This object, which is updated to reflect the trained model
     */
    private AutoencoderBackedSymptomPrevalenceModel initializeAndFit(DataSet dataset, int... dims) throws IOException {
        // Set up a normalizer
        this.normalizer = new NormalizerMinMaxScaler(-1, 1);
        this.normalizer.fit(dataset);
        this.normalizer.transform(dataset);
        ScoredNetworkAndParameters bestModel = null;
        // 5-fold cross validation
        KFoldIterator kfoldIterator = new KFoldIterator(5, dataset);
        // Now do actual training, with hyperparameter selection as follows...
        Random rand = new Random();
        // Keep the seed constant to ensure consistent comparison amongst other hyperparameters
        long seed = Math.round(Math.random() * Long.MAX_VALUE);
        // First select train and test
        for (int fold = 0; fold < 5; fold++) {
            DataSet train = kfoldIterator.next();
            DataSet test = kfoldIterator.testFold();
            for (Activation activationFunction : Activation.values()) { // Activation Function
                for (int i = 0; i < 100; i++) { // 100 samples for l2 regularization between 0.00001 and 0.001
                    double l2 = (rand.nextDouble() * 0.00099) + 0.00001;
                    for (int j = 0; j < 100; j++) { // 100 samples for learning rate between 0.0001 and 0.01
                        double learningRate = (rand.nextDouble() * .0099) + .00001;
                        for (IUpdater optimizationFunc : Arrays.asList(new Sgd(learningRate), new AdaGrad(learningRate))) {
                            ScoredNetworkAndParameters itScore = trainAndScore(train, test, seed, optimizationFunc, learningRate, activationFunction, l2, dims);
                            if (bestModel == null || bestModel.getScores().getMean() > itScore.getScores().getMean()) {
                                bestModel = itScore;
                            }
                        }
                    }
                    // We need to do adaDelta here as it has no learning rate
                    ScoredNetworkAndParameters itScore = trainAndScore(train, test, seed, new AdaDelta(), -1.00, activationFunction, l2, dims);
                    if (bestModel.getScores().getMean() > itScore.getScores().getMean()) {
                        bestModel = itScore;
                    }
                }

            }
        }
        if (bestModel == null) {
            throw new IllegalStateException("No best performing model found somehow... should not be possible");
        }
        this.parameters = bestModel.getHyperparameters();
        this.scores = bestModel.getScores();
        this.model = bestModel.network;
        return this;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private ScoredNetworkAndParameters trainAndScore(DataSet train, DataSet test, long seed, IUpdater optimizationFunc, double learningRate, Activation activationFunction, double l2, int... dims) {
        EarlyStoppingConfiguration earlyStopCondition = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(10, 0.0001))
                .scoreCalculator(new DataSetLossCalculator(new SingletonDataSetIterator(test), true))
                .evaluateEveryNEpochs(1)
                .build();
        MultiLayerNetwork network = getNetworkWithSettings(seed, optimizationFunc, activationFunction, l2, dims);
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlyStopCondition, network, new SingletonDataSetIterator(train));
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        network = result.getBestModel();
        ScoredNetworkAndParameters scoredNetwork = new ScoredNetworkAndParameters(optimizationFunc.getClass().getSimpleName(), learningRate, activationFunction, l2);
        scoredNetwork.setNetwork(network);
        List<DataSet> testData = test.dataSetBatches(1);
        List<Double> scores = new LinkedList<Double>();
        for (DataSet data : testData) {
            scores.add(network.score(data));
        }
        double score = scores.stream().reduce(Double::sum).orElse(0.) / scores.size();
        double stdDev = Math.sqrt(scores.stream().map(d -> Math.pow(d - score, 2)).reduce(Double::sum).orElse(0.) / (scores.size() - 1));
        scoredNetwork.getScores().setMean(score);
        scoredNetwork.getScores().setStdDev(stdDev);
        return scoredNetwork;
    }

    /**
     * Constructs a stacked autoencoder with the provided settings
     *
     * @param seed                 A seed with which to initialize the neural net
     * @param optimizationFunction The optimization function, e.g. SGD
     * @param activation           THe activation function of the neurons, e.g. sigmoid
     * @param l2                   A l2 regularization constant
     * @param dims                 An array of descending dimension sizes for the encoder portion of the autoencoder.
     *                             This will be reversed automatically for the decoder
     * @return A {@link MultiLayerNetwork} representation of the stacked autoencoder matching these settings
     */
    private MultiLayerNetwork getNetworkWithSettings(long seed, IUpdater optimizationFunction, Activation activation, double l2, int... dims) {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(optimizationFunction)
                .activation(activation)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(l2)
                .list();
        int layer = 0;
        if (dims.length < 2) {
            throw new IllegalArgumentException("Must supply at least input dimension and compression dimension size!");
        }
        for (int i = 1; i < dims.length; i++) {
            builder = builder.layer(layer, new DenseLayer.Builder().nIn(dims[i - 1]).nOut(dims[i]).build());
            layer++;
        }
        for (int i = dims.length - 1; i > 1; i--) {
            builder = builder.layer(layer, new DenseLayer.Builder().nIn(dims[i]).nOut(dims[i - 1]).build());
            layer++;
        }
        builder = builder.layer(layer,
                new OutputLayer.Builder().activation(Activation.IDENTITY).nIn(dims[1]).nOut(dims[0])
                        .lossFunction(LossFunctions.LossFunction.MSE).build());

        MultiLayerConfiguration conf = builder.build();
        return new MultiLayerNetwork(conf);
    }


    /**
     * Serializes the model into a specific directory
     * @param modelDirectory A directory to save the model to
     */
    public void serialize(File modelDirectory) throws IOException {
        if (modelDirectory == null || (!modelDirectory.exists() && !modelDirectory.mkdirs()) || !modelDirectory.isDirectory()) {
            throw new IOException("Could not create defined model directory, check it is somewhere writable and does not already exist as a file");
        }
        ObjectWriter ow = new ObjectMapper().writerWithDefaultPrettyPrinter();
        ow.writeValue(new File(modelDirectory, "hyperparameters.json"), parameters);
        ow.writeValue(new File(modelDirectory, "modelstatistics.json"), scores);
        ModelSerializer.writeModel(model, new File(modelDirectory, "model.bin"), true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer();
        normalizerSerializer.write(normalizer, new File(modelDirectory, "normalizer.bin"));
    }

    /**
     * Loads the model from a specific directory
     * @param modelDirectory A directory to load the model from
     */
    private void load(File modelDirectory) throws IOException {
        ObjectMapper om = new ObjectMapper();
        this.parameters = om.readValue(new File(modelDirectory, "hyperparameters.json"), HyperParameters.class);
        this.scores = om.readValue(new File(modelDirectory, "modelstatistics.json"), ModelScore.class);
        this.model = ModelSerializer.restoreMultiLayerNetwork(new File(modelDirectory, "model.bin"), true);
        try {
            this.normalizer = new NormalizerSerializer().restore(new File(modelDirectory, "normalizer.bin"));
        } catch (Exception e) {
            throw new IOException("Failed to load min/max normalizer", e);
        }
    }

    private static class ScoredNetworkAndParameters {
        private HyperParameters hyperparameters;
        private ModelScore scores;
        MultiLayerNetwork network;

        public ScoredNetworkAndParameters() {
        }

        public ScoredNetworkAndParameters(String optimizationFunctionType, double learningRate, Activation activationFunction, double l2) {
            this.hyperparameters = new HyperParameters(optimizationFunctionType, learningRate, activationFunction, l2);
        }

        public ModelScore getScores() {
            return scores;
        }

        public void setScores(ModelScore scores) {
            this.scores = scores;
        }

        public MultiLayerNetwork getNetwork() {
            return network;
        }

        public void setNetwork(MultiLayerNetwork network) {
            this.network = network;
        }

        public HyperParameters getHyperparameters() {
            return hyperparameters;
        }

        public void setHyperparameters(HyperParameters hyperparameters) {
            this.hyperparameters = hyperparameters;
        }
    }

    public static class HyperParameters {
        private String optimizationFunctionType;
        private double learningRate;
        private Activation activationFunction;
        private double l2;

        public HyperParameters() {
        }

        public HyperParameters(String optimizationFunctionType, double learningRate, Activation activationFunction, double l2) {
            this.optimizationFunctionType = optimizationFunctionType;
            this.learningRate = learningRate;
            this.activationFunction = activationFunction;
            this.l2 = l2;
        }

        public String getOptimizationFunctionType() {
            return optimizationFunctionType;
        }

        public void setOptimizationFunctionType(String optimizationFunctionType) {
            this.optimizationFunctionType = optimizationFunctionType;
        }

        public double getLearningRate() {
            return learningRate;
        }

        public void setLearningRate(double learningRate) {
            this.learningRate = learningRate;
        }

        public Activation getActivationFunction() {
            return activationFunction;
        }

        public void setActivationFunction(Activation activationFunction) {
            this.activationFunction = activationFunction;
        }

        public double getL2() {
            return l2;
        }

        public void setL2(double l2) {
            this.l2 = l2;
        }
    }

    public static class ModelScore {
        private double mean;
        private double stdDev;

        public ModelScore() {
        }

        public ModelScore(double mean, double stdDev) {
            this.mean = mean;
            this.stdDev = stdDev;
        }

        public double getMean() {
            return mean;
        }

        public void setMean(double mean) {
            this.mean = mean;
        }

        public double getStdDev() {
            return stdDev;
        }

        public void setStdDev(double stdDev) {
            this.stdDev = stdDev;
        }
    }

    public HyperParameters getParameters() {
        return parameters;
    }

    public ModelScore getScores() {
        return scores;
    }

    public NormalizerMinMaxScaler getNormalizer() {
        return normalizer;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

}
