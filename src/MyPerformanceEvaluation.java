import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.Iterator;

/**
 * Created by DELL on 2017/3/27.
 */
public class MyPerformanceEvaluation {
    private static final int pointNum = 200;
    private double TP;
    private double TN;
    private double FP;
    private double FN;
    private double threshold;
    private Formatter RocOut;
    private double minLabel;
    private double maxLabel;
    private double labelStep;
    private ArrayList<MyUtil.Record> testSet;
    private ArrayList<Double> basicRes;
    private ArrayList<double[][]> minMaxModuleRes;

    void setTestSet(ArrayList<MyUtil.Record> testSet) {
        this.testSet = testSet;
    }

    private void initIndeces() {
        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;
    }

    private void init(String filename, double minLabel, double maxLabel, double threshold) throws IOException {
        this.minLabel = minLabel;
        this.maxLabel = maxLabel;
        this.threshold = threshold;
        RocOut = new Formatter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename + " ROC"), Linear.FILE_CHARSET)));
        this.labelStep = (maxLabel-minLabel)/pointNum;
        initIndeces();
    }

    void initBasic(ArrayList<Double> result, String filename, double minLabel, double maxLabel, double threshold) throws IOException {
        init(filename, minLabel, maxLabel, threshold);
        basicRes = result;
    }

    void initMinMaxModule(ArrayList<double[][]> result, String filename, double minLabel, double maxLabel, double threshold) throws IOException {
        init(filename, minLabel, maxLabel, threshold);
        minMaxModuleRes = result;
    }

    private void processBasicROC(double standard, double threshold) throws IOException {
        Iterator<Double> iterator = basicRes.iterator();
        Iterator<MyUtil.Record> setIterator = testSet.iterator();
        initIndeces();
        while (iterator.hasNext()) {
            double rawLabel = iterator.next();
            double realLabel = setIterator.next().label;
            updateIndeces(realLabel, rawLabel, standard, threshold);
        }
    }

    private void processMinMaxModuleROC(double standard, double threshold) throws IOException {
        Iterator<double[][]> iterator = minMaxModuleRes.iterator();
        Iterator<MyUtil.Record> setIterator = testSet.iterator();
        initIndeces();
        while (iterator.hasNext()) {
            double[][] rawLabels = iterator.next();
            double predictedLabel = MyUtil.MinMaxModule(rawLabels, standard, threshold);
            double realLabel = setIterator.next().label;
            updateIndeces(realLabel, predictedLabel, 0, 0);
        }
    }

    private void updateIndeces(double realLabel, double predictedLabel, double standard, double threshold) throws IOException {
        if (realLabel > 0) {
            if (predictedLabel > standard + threshold)
                TP++;
            if (predictedLabel < standard - threshold)
                FN++;
        } else {
            if (predictedLabel > standard + threshold)
                FP++;
            if (predictedLabel < standard - threshold)
                TN++;
        }
    }

    private void recordROCCoordinates() throws IOException {
        double FPR = FP/(FP+TN);
        double TPR = TP/(TP+FN);
        Linear.printf(RocOut, "%g\t%g\n", FPR, TPR);
    }

    private void printResult() {
        double accuracy = (TP+TN)/(TP+TN+FP+FN);
        double precision = TP/(TP+FP);
        double recall = TP/(TP+FN);
        double F1 = 2*recall*precision/(recall+precision);
        Linear.info("Evaluation Result:\nAccuracy: %g\nPrecision: %g\nRecall: %g\nF1: %g\n", accuracy, precision, recall, F1);
    }

    void basicEvaluation() throws IOException {
        Linear.info("Basic Evaluation Begin...\n");
        double standard = minLabel;
        while (standard <= maxLabel) {
            processBasicROC(standard, 0);
            recordROCCoordinates();
            standard += labelStep;
        }
        Linear.closeQuietly(RocOut);
        processBasicROC(0, threshold);
        printResult();
        Linear.info("Basic Evaluation End\n\n\n");
    }

    void minMaxModuleEvaluation() throws IOException {
        Linear.info("Min_Max_Module Evaluation Begin...\n");
        double standard = minLabel;
        while (standard <= maxLabel) {
            processMinMaxModuleROC(standard, 0);
            recordROCCoordinates();
            standard += labelStep;
        }
        Linear.closeQuietly(RocOut);
        processMinMaxModuleROC(0, threshold);
        printResult();
        Linear.info("Min_Max_Module Evaluation End\n\n\n");
    }
}
