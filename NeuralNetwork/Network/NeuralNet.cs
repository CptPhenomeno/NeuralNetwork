namespace NeuralNetwork.Network
{
    using System;
    using System.Collections;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    using NeuralNetwork.Layer;
    using NeuralNetwork.Activation;

    public class NeuralNet :IEnumerable
    {
        private Layer[] netLayers;
        private Vector<double> input;
        private Vector<double> output;

        public NeuralNet(int numOfInput, int[] sizeOfLayers, IActivationFunction[] activationOfLayers)
        {
            if (sizeOfLayers.Length != activationOfLayers.Length)
                throw new ArgumentException("Mismatch from number of layers and number of activations functions");

            int length = sizeOfLayers.Length;
            netLayers = new Layer[length];

            netLayers[0] = new Layer(activationOfLayers[0], sizeOfLayers[0], numOfInput);

            for (int index = 1; index < length; index++)
            {
                netLayers[index] = new Layer(activationOfLayers[index],sizeOfLayers[index], sizeOfLayers[index-1]);
            }

            output = Vector<double>.Build.Dense(sizeOfLayers[length - 1], 1);
        }

        public void ComputeOutput(double[] x)
        {
            ComputeOutput(Vector<double>.Build.DenseOfArray(x));
        }

        public void ComputeOutput(Vector<double> x)
        {
            Input = x;
            Vector<double> tmp = x;

            foreach (Layer layer in netLayers)
            {
                layer.ComputeOutput(tmp);
                tmp = layer.Output;
            }

            tmp.CopyTo(output);
        }

        public void UpdateNetwork(Matrix<double>[] weightsUpdates, Vector<double>[] biasesUpdates)
        {
            for (int nextLayerIndex = 0; nextLayerIndex < NumberOfLayers; nextLayerIndex++)
            {
                netLayers[nextLayerIndex].Update(weightsUpdates[nextLayerIndex], biasesUpdates[nextLayerIndex]);
            }
        }

        #region Getter & Setter

        public Layer this[int layerIndex]
        {
            get { return netLayers[layerIndex]; }
        }

        public Vector<double> Input
        {
            get { return input; }
            set { input = value; }
        }

        public Vector<double> Output
        {
            get { return output; }
        }

        public int NumberOfLayers
        {
            get { return netLayers.Length; }
        }

        public Layer OutputLayer
        {
            get { return netLayers[NumberOfLayers - 1]; }
        }

        #endregion

        public void RandomizeWeights()
        {
            foreach (Layer l in netLayers)
                l.RandomizeWeights();
        }

        public IEnumerator GetEnumerator()
        {
            for (int layerIndex = 0; layerIndex < NumberOfLayers; layerIndex++)
                yield return netLayers[layerIndex];
        }
    }
}
