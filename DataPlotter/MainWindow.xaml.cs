using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using System.Drawing;

namespace DataPlotter
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            AddChartToMainWindow();
        }

        private void AddChartToMainWindow()
        {
            Chart neuralNetErrorChart = new Chart();

            //Series is where the points are "stored"
            Series netErrorThroughEpochs = new Series();
            netErrorThroughEpochs.ChartType = SeriesChartType.Line;
            netErrorThroughEpochs.Color = Color.DodgerBlue;

            //Represent the drawing area
            ChartArea chartArea = new ChartArea();
            chartArea.AxisX.Title = "Epoch";
            chartArea.AxisY.Title = "Net Error (MSE)";

            neuralNetErrorChart.ChartAreas.Add(chartArea);
            neuralNetErrorChart.Series.Add(netErrorThroughEpochs);

            winformhost.Child = neuralNetErrorChart;
        }

    }
}
