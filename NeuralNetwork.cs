using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork
{
    public Layer[] layers;
    public float momentum;
    public float learningRate;
    public int numberOfLayers;

    public float fitness = 0;

    public NeuralNetwork(int[] topology, float learningRate, float momentum)
    {
        this.learningRate = learningRate;
        this.momentum = momentum;
        numberOfLayers = topology.Length;
        layers = new Layer[numberOfLayers];

        layers[0] = new Layer(topology[0], topology[0]);

        for (int i = 1; i < numberOfLayers; i++)
        {
            layers[i] = new Layer(topology[i], topology[i - 1]);

            //Debug.Log(layers[i].input);
        }
    }

    public void feedForward()
    {
        for (int i = 0; i < layers[0].neurons.Length; i++)
        {
            layers[0].neurons[i].output = layers[0].input[i];
        }

        layers[1].input = layers[0].input;
        for (int i = 1; i < numberOfLayers; i++)
        {
            layers[i].feedForward();

            if (i != numberOfLayers - 1)
            {
                layers[i + 1].input = layers[i].getOutputs();
            }
        }
    }

    private void backpropagateError()
    {
        for (int i = numberOfLayers - 1; i > 0; i--)
        {
            for (int j = 0; j < layers[i].neurons.Length; j++)
            {
                // Calculate bias difference
                layers[i].neurons[j].biasDiff =
                        learningRate *
                        layers[i].neurons[j].signalError +
                        momentum *
                        layers[i].neurons[j].biasDiff;

                // Update bias
                layers[i].neurons[j].bias += layers[i].neurons[j].biasDiff;

                // Update weights
                for (int k = 0; k < layers[i].input.Length; k++)
                {
                    // Calculate weight difference
                    layers[i].neurons[j].weightsDiff[k] =
                            learningRate *
                            layers[i].neurons[j].signalError *
                            layers[i - 1].neurons[k].output +
                            momentum *
                            layers[i].neurons[j].weightsDiff[k];

                    // Update weight
                    layers[i].neurons[j].weights[k] += layers[i].neurons[j].weightsDiff[k];
                }
            }
        }
    }


    private void calculateSignalErrors(float[] expectedOutputs)
    {
        int outputLayer = numberOfLayers - 1;
        for (int i = 0; i < layers[outputLayer].neurons.Length; i++)
        {
            layers[outputLayer].neurons[i].signalError =
                    (expectedOutputs[i] -
                    layers[outputLayer].neurons[i].output) *
                    layers[outputLayer].neurons[i].output *
                    (1 - layers[outputLayer].neurons[i].output);
        }

        float tempSum = 0;
        for (int i = numberOfLayers - 2; i > 0; i--)
        {
            for (int j = 0; j < layers[i].neurons.Length; j++)
            {
                tempSum = 0;
                for (int k = 0; k < layers[i + 1].neurons.Length; k++)
                {
                    tempSum +=
                            layers[i + 1].neurons[k].weights[j] *
                            layers[i + 1].neurons[k].signalError;
                }

                layers[i].neurons[j].signalError =
                        layers[i].neurons[j].output *
                        (1 - layers[i].neurons[j].output) *
                        tempSum;
            }
        }
    }

    private void updateWeights(float[] expectedOutputs)
    {
        calculateSignalErrors(expectedOutputs);
        backpropagateError();
    }

    public void train(float[] inputs, float[] expectedOutputs, bool print)
    {
        for (int i = 0; i < layers[0].neurons.Length; i++)
        {
            layers[0].input[i] = inputs[i];
        }

        feedForward();
        updateWeights(expectedOutputs);

    }


    public float getOverallError(float[] expectedOutputs)
    {
        float totalError = 0;
        for (int i = 0; i < layers[numberOfLayers - 1].neurons.Length; i++)
        {
            totalError += 0.5f * Mathf.Pow(expectedOutputs[i] - layers[numberOfLayers - 1].neurons[i].output, 2);
        }
        return totalError;
    }

    public float[] predict(float[] inputs)
    {
        for (int i = 0; i < layers[0].neurons.Length; i++)
        {
            layers[0].input[i] = inputs[i];
        }

        feedForward();

        return layers[numberOfLayers - 1].getOutputs();
    }

    public int getOutputSize()
    {
        return layers[numberOfLayers - 1].getNumberOfNeurons();
    }
}
