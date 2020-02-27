using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public float[] input;
    public Neuron[] neurons;

    public Layer(int numberOfNeurons, int numberOfInputs)
    {
        neurons = new Neuron[numberOfNeurons];
        input = new float[numberOfInputs];

        for (int i = 0; i < numberOfNeurons; i++)
        {
            neurons[i] = new Neuron(numberOfInputs);
        }
    }

    public void feedForward()
    {
        float tempOutput;

        for (int i = 0; i < neurons.Length; i++)
        {
            tempOutput = neurons[i].bias;

            for (int j = 0; j < neurons[i].weights.Length; j++)
            {
                tempOutput += input[j] * neurons[i].weights[j];
            }

            neurons[i].output = Layer.sigmoid(tempOutput);
        }
    }

    public float[] getOutputs()
    {
        float[] outputs = new float[neurons.Length];

        for (int i = 0; i < outputs.Length; i++)
        {
            outputs[i] = neurons[i].output;
        }

        return outputs;
    }


    public static float sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp((-x)));
    }

    public int getNumberOfNeurons()
    {
        return neurons.Length;
    }
}
