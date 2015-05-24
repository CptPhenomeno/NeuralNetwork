namespace NeuralNetwork.Learning
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    using NeuralNetwork.Network;
    using NeuralNetwork.Layer;

    using DatasetUtility;

    public class BackPropagation
    {
        private NeuralNet net;
        private Vector<double>[] deltas;
        private Matrix<double>[] weightsUpdates;
        private Matrix<double>[] oldWeightsUpdates;
        private Vector<double>[] biasesUpdates;
        private int batchSize;
        private double learningRate;
        private double momentum;

        public BackPropagation(NeuralNet net, double learningRate, double momentum, int batchSize)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.batchSize = batchSize;

            this.net = net;
            InitializeNetworkValues();
        }

        public double Run(int start, Vector<double>[] inputs, Vector<double>[] expectedOutputs)
        {
            double error = 0.0;

            for (int next = start; next < start + batchSize; next++)
            {
                Vector<double> input = inputs[next];
                net.Input = input;
                net.ComputeOutput(input);
                //Vector with error for each output of the network
                Vector<double> netError = expectedOutputs[next] - net.Output;
                //I think that this matrix is 1x1 but is better check...
                error += netError.DotProduct(netError);

                ComputeOutputLayerUpdate(netError);
                ComputeHiddenLayersUpdate(input);                
            }

            return error;
        }

        private void ComputeOutputLayerUpdate(Vector<double> netError)
        {
            int outputLayerIndex = net.NumberOfLayers - 1;
            //Delta factor of output layer using Hadamard multiplication
            deltas[outputLayerIndex] = netError.PointwiseMultiply(net[outputLayerIndex].LocalFieldDifferentiated);            

            //Update value for output biases
            biasesUpdates[outputLayerIndex].Add(deltas[outputLayerIndex].Multiply(learningRate), biasesUpdates[outputLayerIndex]);

            Vector<double> outputLayerInput = null;
            if (net.NumberOfLayers > 1)
                outputLayerInput = net[outputLayerIndex - 1].Output;
            else if (net.NumberOfLayers == 1)
                outputLayerInput = net.Input;
            
            //Update value for output weights
            //I have some doubt here...
            Matrix<double> update = biasesUpdates[outputLayerIndex].ToColumnMatrix().Multiply(outputLayerInput.ToRowMatrix());
            weightsUpdates[outputLayerIndex].Add(update, weightsUpdates[outputLayerIndex]);
            weightsUpdates[outputLayerIndex].Add(oldWeightsUpdates[outputLayerIndex], weightsUpdates[outputLayerIndex]);
            oldWeightsUpdates[outputLayerIndex].Add(update.Multiply(momentum),oldWeightsUpdates[outputLayerIndex]);
        }

        private void ComputeHiddenLayersUpdate(Vector<double> input)
        {
            int actualLayerIndex = net.NumberOfLayers - 2;
            int numOfLayers = net.NumberOfLayers;

            for (; actualLayerIndex >= 0; actualLayerIndex--)
            {                
                //The next layer support for update
                Vector<double> sigma = net[actualLayerIndex + 1].Weights.Transpose().Multiply(deltas[actualLayerIndex + 1]);
                //Vector<double> sigma = weightsUpdates[actualLayerIndex + 1].Transpose().Multiply(deltas[actualLayerIndex + 1]);
                deltas[actualLayerIndex] = sigma.PointwiseMultiply(net[actualLayerIndex].LocalFieldDifferentiated);

                biasesUpdates[actualLayerIndex].Add(deltas[actualLayerIndex].Multiply(learningRate), biasesUpdates[actualLayerIndex]);

                Vector<double> previousLayerOutput = 
                    (actualLayerIndex == 0) ? input : net[actualLayerIndex - 1].Output;
                
                Matrix<double> update = biasesUpdates[actualLayerIndex].ToColumnMatrix().Multiply(previousLayerOutput.ToRowMatrix());
                weightsUpdates[actualLayerIndex].Add(update,weightsUpdates[actualLayerIndex]);
                weightsUpdates[actualLayerIndex].Add(oldWeightsUpdates[actualLayerIndex], weightsUpdates[actualLayerIndex]);
                oldWeightsUpdates[actualLayerIndex].Add(update.Multiply(momentum), oldWeightsUpdates[actualLayerIndex]);
            }
            
        }

        public void UpdateNetwork()
        {
            if (batchSize > 1)
            {
                foreach (Matrix<double> m in weightsUpdates)
                    m.Divide(batchSize, m);
                foreach (Vector<double> b in biasesUpdates)
                    b.Divide(batchSize, b);
            }

            net.UpdateNetwork(weightsUpdates, biasesUpdates);
            foreach (Matrix<double> m in weightsUpdates)
                m.Clear();
            foreach (Vector<double> b in biasesUpdates)
                b.Clear();
            foreach (Matrix<double> old in oldWeightsUpdates)
                old.Clear();
        }

        #region Getter & Setter

        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value; }
        }

        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public double Momentum
        {
            get { return momentum; }
            set { momentum = value; }
        }

        #endregion

        private void InitializeNetworkValues()
        {
            int numberOfLayers = net.NumberOfLayers;

            deltas = new Vector<double>[numberOfLayers];
            weightsUpdates = new Matrix<double>[numberOfLayers];
            oldWeightsUpdates = new Matrix<double>[numberOfLayers];
            biasesUpdates = new Vector<double>[numberOfLayers];

            int nextLayer = 0;
            foreach (Layer layer in net)
            {
                deltas[nextLayer] = Vector<double>.Build.Dense(layer.NumberOfNeurons);
                weightsUpdates[nextLayer] = Matrix<double>.Build.Dense(layer.NumberOfNeurons, layer.NumberOfInputs);
                oldWeightsUpdates[nextLayer] = Matrix<double>.Build.Dense(layer.NumberOfNeurons, layer.NumberOfInputs);
                biasesUpdates[nextLayer] = Vector<double>.Build.Dense(layer.NumberOfNeurons);
                nextLayer++;
            }
        }
    }
}
