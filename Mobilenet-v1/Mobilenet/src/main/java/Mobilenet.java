import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class Mobilenet {
    private static Logger log = LoggerFactory.getLogger(Mobilenet.class);
    public final static int random = 123;

    public static void main(String[] args) throws IOException {
        //Training Set and Test Set
        MnistDataSetIterator train = new MnistDataSetIterator(32, true, random);
        MnistDataSetIterator test = new MnistDataSetIterator(32, false, random);

        //Setup the configurations
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(random)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .l2(0.0001)
                .list()
                .layer(0, new Convolution2D.Builder()
                        .kernelSize(3,3)
                        .stride(2,2)
                        .nOut(32)
                        .build())
                .layer (1, new BatchNormalization.Builder()
                        .build())
                .layer(2, new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DepthwiseConvolution2D.Builder()
                        .kernelSize(3,3)
                        .depthMultiplier(1)
                        .nOut(32)
                        .build())
                .layer(4, new BatchNormalization.Builder()
                        .build())
                .layer(5, new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new Convolution2D.Builder()
                        .kernelSize(1,1)
                        .nOut(64)
                        .build())
                .layer(7, new BatchNormalization.Builder()
                        .build())
                .layer(8, new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(9, new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build()
                        )
                .layer(10, new DenseLayer.Builder()
                        .nOut(100)
                        .build()
                )
                .layer(11, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nOut(10)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28, 1))
                .build();

        //Initialize the neural network
        MultiLayerNetwork nn = new MultiLayerNetwork(config);
        nn.init();

        //Train
        nn.setListeners(new ScoreIterationListener(10));
        log.info("Training model");
        nn.fit(train, 5);
        log.info("Evaluating model");
        Evaluation eval = nn.evaluate(test);
        log.info(eval.stats());

    }
}
