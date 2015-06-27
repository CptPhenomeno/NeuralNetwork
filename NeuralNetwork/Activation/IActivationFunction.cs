namespace NeuralNetwork.Activation
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public interface IActivationFunction
    {
        Vector<double> Function(Vector<double> x);
        Vector<double> Derivative(Vector<double> x);
    }
}
