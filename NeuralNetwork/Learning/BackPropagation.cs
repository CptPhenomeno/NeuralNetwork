namespace NeuralNetwork.Learning
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    using NeuralNetwork.Network;
    using NeuralNetwork.Layer;

    using DatasetUtility;

    public class Backpropagation
    {
        private NeuralNet net;
        private Vector<double>[] deltas;
        private Matrix<double>[] weightsUpdates;
        private Matrix<double>[] oldWeightsUpdates;
        private Vector<double>[] biasesUpdates;
        
        private int batchSize;

        private double learningRate;
        private double momentum;
        private double weightDecay;

        private int samplesUsed;

        public Backpropagation(NeuralNet net, double learningRate, double momentum, double weightDecay, int batchSize)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.batchSize = batchSize;
            this.weightDecay = weightDecay;

            this.net = net;
            InitializeNetworkValues();

            samplesUsed = 0;
        }

		public double Run(Sample sample)
        {
            ++samplesUsed;

            double error = 0.0;

            net.ComputeOutput(sample.Input);

            //Vector with error for each output of the network
            Vector<double> netError = sample.Output - net.Output;

            error += netError.DotProduct(netError);
            error /= 2;

            ComputeOutputLayerUpdate(netError);
            ComputeHiddenLayersUpdate(sample.Input);

            return error;
        }

        private void ComputeOutputLayerUpdate(Vector<double> netError)
        {
            int outputLayerIndex = net.NumberOfLayers - 1;
            //Delta factor of output layer using Hadamard multiplication
            deltas[outputLayerIndex] = netError.PointwiseMultiply(net[outputLayerIndex].LocalFieldDifferentiated);            

            //Update value for output biases
            //biasesUpdates[outputLayerIndex].Add(deltas[outputLayerIndex].Multiply(learningRate), biasesUpdates[outputLayerIndex]);
            biasesUpdates[outputLayerIndex].Add(deltas[outputLayerIndex], biasesUpdates[outputLayerIndex]);

            //Assume that the network have at least two layers
            Vector<double> outputLayerInput = net[outputLayerIndex - 1].Output;
            
            //Update value for output weights
            //I have some doubt here...
            Matrix<double> weightDecayMatrix = net[outputLayerIndex].Weights.Multiply(2).Multiply(WeightDecay);

            Matrix<double> update = deltas[outputLayerIndex].ToColumnMatrix().Multiply(outputLayerInput.ToRowMatrix());
            weightsUpdates[outputLayerIndex].Add(update, weightsUpdates[outputLayerIndex]);
            weightsUpdates[outputLayerIndex].Add(oldWeightsUpdates[outputLayerIndex], weightsUpdates[outputLayerIndex]);
            oldWeightsUpdates[outputLayerIndex].Add(update.Multiply(Momentum), oldWeightsUpdates[outputLayerIndex]);

            weightsUpdates[outputLayerIndex].Add(weightDecayMatrix, weightsUpdates[outputLayerIndex]);
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

                biasesUpdates[actualLayerIndex].Add(deltas[actualLayerIndex], biasesUpdates[actualLayerIndex]);

                Vector<double> previousLayerOutput = 
                    (actualLayerIndex == 0) ? input : net[actualLayerIndex - 1].Output;

                Matrix<double> weightDecayMatrix = net[actualLayerIndex].Weights.Multiply(2).Multiply(WeightDecay);
                
                Matrix<double> update = deltas[actualLayerIndex].ToColumnMatrix().Multiply(previousLayerOutput.ToRowMatrix());
                weightsUpdates[actualLayerIndex].Add(update,weightsUpdates[actualLayerIndex]);
                weightsUpdates[actualLayerIndex].Add(oldWeightsUpdates[actualLayerIndex], weightsUpdates[actualLayerIndex]);
                oldWeightsUpdates[actualLayerIndex].Add(update.Multiply(Momentum), oldWeightsUpdates[actualLayerIndex]);

                weightsUpdates[actualLayerIndex].Add(weightDecayMatrix, weightsUpdates[actualLayerIndex]);
            }
            
        }

        public void UpdateNetwork()
        {
			foreach (Matrix<double> m in weightsUpdates)
            {
                m.Multiply(LearningRate, m);
                m.Divide(samplesUsed, m);
            }

            foreach (Vector<double> b in biasesUpdates)
            {
                b.Multiply(LearningRate, b);
                b.Divide(samplesUsed, b);
            }
			
            net.UpdateNetwork(weightsUpdates, biasesUpdates);
            foreach (Matrix<double> m in weightsUpdates)
                m.Clear();
            foreach (Vector<double> b in biasesUpdates)
                b.Clear();
            foreach (Matrix<double> old in oldWeightsUpdates)
                old.Clear();

            samplesUsed = 0;
        }

        #region Getter & Setter

        public NeuralNet Net
        {
            get { return net; }
            set { net = value; }
        }

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

        public double WeightDecay
        {
            get { return weightDecay; }
            set { weightDecay = value; }
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
