namespace NeuralNetwork.Learning
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    using NeuralNetwork.Network;
    using NeuralNetwork.Layer;

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
        private double maxError;
        private int maxEpoch;

        public BackPropagation(NeuralNet net, double learningRate = 0.3, double momentum = 0.0, double maxError = 0.01,
                               int maxEpoch = 1000, int batchSize = 1)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.maxError = maxError;
            this.maxEpoch = maxEpoch;
            this.batchSize = batchSize;

            this.net = net;
            InitializeNetworkValues();
        }

        public void Learn(double[][] inputs, double[][] outputs, double[][] testSet = null)
        {
            Vector<double>[] vectorInputs = new Vector<double>[inputs.Length];
            Vector<double>[] vectorOutputs = new Vector<double>[outputs.Length];
            Vector<double>[] vectorTest = null;

            int inputIndex = 0, outputIndex = 0;

            foreach (double[] input in inputs)
                vectorInputs[inputIndex++] = Vector<double>.Build.DenseOfArray(input);

            foreach (double[] output in outputs)
                vectorOutputs[outputIndex++] = Vector<double>.Build.DenseOfArray(output);

            if (testSet != null)
            {
                int testIndex = 0;
                foreach (double[] test in testSet)
                    vectorOutputs[testIndex++] = Vector<double>.Build.DenseOfArray(test);
            }

            Learn(vectorInputs, vectorOutputs, vectorTest);
        }

        public void Learn(Vector<double>[] inputs, Vector<double>[] expectedOutputs, Vector<double>[] testSet = null)
        {
            double error = 0.0;
            int epoch = -1;
            int numOfElemInEpoch = inputs.Length;

            do
            {
                ++epoch;
                error = RunEpoch(inputs, expectedOutputs);
                error /= numOfElemInEpoch;                
            } while (error > MaxError && epoch < MaxEpoch);
            Console.WriteLine("Error {0} at epoch {1}", error, epoch);
        }

        private double RunEpoch(Vector<double>[] inputs, Vector<double>[] expectedOutputs)
        {
            double error = 0.0;

            for (int next = 0; next < inputs.Length; next += batchSize)
            {
                error += RunBatch(next, inputs, expectedOutputs);
                //Update the network only after batchSize examples
                UpdateNetwork();
            }

            return error;
        }

        private double RunBatch(int start, Vector<double>[] inputs, Vector<double>[] expectedOutputs)
        {
            double error = 0.0;

            for (int next = start; next < start + batchSize; next++)
            {
                Vector<double> input = inputs[next];
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

            Vector<double> outputLayerInput = net[outputLayerIndex - 1].Output;
            
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

        private void UpdateNetwork()
        {
            foreach (Matrix<double> m in weightsUpdates)
                m.Divide(batchSize, m);
            foreach (Vector<double> b in biasesUpdates)
                b.Divide(batchSize, b);

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

        public double MaxError
        {
            get { return maxError; }
            set { maxError = value; }
        }

        public int MaxEpoch
        {
            get { return maxEpoch; }
            set { maxEpoch = value; }
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
