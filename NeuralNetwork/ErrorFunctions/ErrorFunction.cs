namespace NeuralNetwork.ErrorFunctions
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public delegate double ErrorFunction(Vector<double> targetValues, Vector<double> outputValues);
}
