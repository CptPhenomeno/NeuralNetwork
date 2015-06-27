namespace NeuralNetwork.Activation
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class LinearFunction : IActivationFunction
    {
        public Vector<double> Function(Vector<double> x)
        {
            return x;
        }

        public Vector<double> Derivative(Vector<double> x)
        {
            return Vector<double>.Build.Dense(x.Count, 1.0);
        }
    }
}
