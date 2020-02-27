using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public float output;
    public float[] weights;
    public float[] weightsDiff;
    public float bias;
    public float biasDiff;
    public float signalError;

    public Neuron(int numberOfWeights)
    {
        bias = -1 + Random.Range(0.0f, 1.0f) * 2;
        //bias = Random.Range(0, 1);
        biasDiff = 0;


        weights = new float[numberOfWeights];
        weightsDiff = new float[numberOfWeights];

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = -1 + Random.Range(0.0f, 1.0f) * 2;
            //  weights[i] = Random.Range(0, 1);
            weightsDiff[i] = 0;
        }

    }
}
