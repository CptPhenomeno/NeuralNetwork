namespace NeuralNetwork.Layer
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.Distributions;

    using NeuralNetwork.Activation;
    
    /*
    delegate double ActivationFunction(double x);
    delegate double ActivationFunctionDerivative(double x);
    */
    
    public class Layer
    {
        private Matrix<double> weights;
        private Matrix<double> bias;
        private Matrix<double> localField;
        private Matrix<double> output;
        private IActivationFunction activationFunction;
        private int numberOfNeurons;
        private int numberOfInputs;

        public Layer(IActivationFunction activationFunction, int numOfNeurons, int numOfInputs)
        {
            this.activationFunction = activationFunction;
            bias = Matrix<double>.Build.Random(numOfNeurons, 1, new Normal(0, 0.5));
            localField = Matrix<double>.Build.Dense(numOfNeurons, 1);
            output = Matrix<double>.Build.Dense(numOfNeurons, 1);
            weights = Matrix<double>.Build.Random(numOfNeurons, numOfInputs, new Normal(0, 0.5));
            this.numberOfNeurons = numOfNeurons;
            this.numberOfInputs = numOfInputs;
        }

        public void ComputeOutput(Matrix<double> input)
        {
            //Compute the local field without the bias
            weights.Multiply(input, localField);
            //Add the bias for the real induced local field
            localField.Add(bias, localField);
            //Apply the activation function to the local field
            localField.Map((l => activationFunction.Function(l)), output);
        }

        public void Update(Matrix<double> weightsUpdates, Matrix<double> biasesUpdates)
        {
            weights.Add(weightsUpdates, weights);
            bias.Add(biasesUpdates, bias);
        }

        #region Getter & Setter

        public Matrix<double> Weights
        {
            get { return weights; }
        }

        public Matrix<double> Output
        {
            get { return output; }
        }

        public Matrix<double> LocalFieldDifferentiated
        {
            get { return localField.Map((l => activationFunction.Derivative(l))); }
        }

        public int NumberOfNeurons
        {
            get { return numberOfNeurons; }
        }

        public int NumberOfInputs
        {
            get { return numberOfInputs; }
        }

        #endregion
    }
}
