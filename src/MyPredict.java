import com.sun.org.apache.regexp.internal.RE;

import java.io.*;
import java.util.*;

/**
 * Created by DELL on 2017/3/25.
 */
public class MyPredict {
    private static final double threshold = 0;
    private static int nr_feature;
    private static double bias;
    private static ArrayList<MyUtil.Record> testSet;
    private static ArrayList<Double> basicRawLabels;
    private static ArrayList<double[][]> minMaxModuleRawLabels;
    private static double minRawLabel;
    private static double maxRawLabel;
    private static MyPerformanceEvaluation evaluation;

    MyPredict(String inputFilename, int nr_feature, double bias) throws IOException, InvalidInputDataException {
        MyPredict.nr_feature = nr_feature;
        MyPredict.bias = bias;
        testSet = new ArrayList<>();
        evaluation = new MyPerformanceEvaluation();
        readDataFile(new File(inputFilename));
    }

    private static void readDataFile(File file) throws IOException, InvalidInputDataException {
        int lineNr = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(file))){
            while (true) {
                String line = br.readLine();
                if (line == null)
                    break;
                lineNr++;

                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                String token;
                try {
                    token = st.nextToken();
                } catch (NoSuchElementException e) {
                    throw new InvalidInputDataException("empty line", file, lineNr, e);
                }

                double label;
                try {
                    label = MyUtil.transferLabel(token);
                } catch (NumberFormatException e) {
                    throw new InvalidInputDataException("invalid label: " + token, file, lineNr, e);
                }

                int m = st.countTokens()/2;
                Feature[] features;
                if (bias >= 0) {
                    features = new Feature[m+1];
                    features[m] = new FeatureNode(nr_feature+1, bias);
                } else {
                    features = new Feature[m];
                }
                int oldIndex = 0;
                for (int i = 0; i < m; i++) {
                    token = st.nextToken();
                    int index;
                    try {
                        index = Linear.atoi(token);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid index: " + token, file, lineNr, e);
                    }

                    if (index < 0)
                        throw new InvalidInputDataException("invalid index: " + index, file, lineNr);
                    if (index <= oldIndex)
                        throw new InvalidInputDataException("indices must be sorted in ascending order", file, lineNr);
                    oldIndex = index;

                    token = st.nextToken();
                    try {
                        double value = Linear.atof(token);
                        features[i] = new FeatureNode(index, value);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid value: " + token, file, lineNr);
                    }
                }
                testSet.add(new MyUtil.Record(label, features));
            }
            evaluation.setTestSet(testSet);
            Linear.info("Test Set Size: %d\n", lineNr);
        }
    }

    private static void updateMinMaxModuleRawLabel(double rawLabel) {
        minRawLabel = Math.min(minRawLabel, rawLabel);
        maxRawLabel = Math.max(maxRawLabel, rawLabel);
    }

    private static void doBasicPredict(Model model, Writer writer) throws IOException {
        int correct = 0;
        int total = 0;
        double error = 0;
        double sump = 0, sumr = 0, sumpp = 0, sumrr = 0, sumpr = 0;
        Formatter out = new Formatter(writer);
        Iterator<MyUtil.Record> iterator = testSet.iterator();
        MyUtil.Record record;

        while(iterator.hasNext()) {
            record = iterator.next();
            double realLabel = record.label;
            Feature[] nodes = record.features;
            double rawLabel, predictedLabel;

            rawLabel = Linear.patentPredictValue(model, nodes);
            updateMinMaxModuleRawLabel(rawLabel);
            basicRawLabels.add(rawLabel);
            predictedLabel = rawLabel>threshold ? 1 : ((rawLabel<-threshold) ? -1 : 0);
            Linear.printf(out, "%g\n", predictedLabel);
            if (predictedLabel == realLabel)
                ++correct;
            error += (predictedLabel-realLabel)*(predictedLabel-realLabel);
            sump += predictedLabel;
            sumr += realLabel;
            sumpp += predictedLabel*predictedLabel;
            sumrr += realLabel*realLabel;
            sumpr += predictedLabel*realLabel;
            ++total;
        }
        if (model.solverType.isSupportVectorRegression()) {
            Linear.info("Mean squared error = %g (regression)%n", error/total);
            Linear.info("Squared correlation coefficient = %g (regression)%n", ((total*sumpr-sump*sumr)*(total*sumpr-sump*sumr))/((total*sumpp-sump*sump)*(total*sumrr-sumr*sumr)));
        } else {
            Linear.info("Accuracy = %g%% (%d/%d)%n", (double)correct/total*100, correct, total);
        }
    }

    private static void doMinMaxModulePredict(Model[][] models, Writer writer) throws IOException {
        int correct = 0;
        int total = 0;
        double error = 0;
        double sump = 0, sumr = 0, sumpp = 0, sumrr = 0, sumpr = 0;
        Formatter out = new Formatter(writer);
        Iterator<MyUtil.Record> iterator = testSet.iterator();
        int modelM = models.length;
        int modelN = models[0].length;
        MyUtil.Record record;
        double[][] labels;

        while (iterator.hasNext()) {
            record = iterator.next();
            double realLabel = record.label;
            Feature[] nodes = record.features;
            double predictedLabel;
            labels = new double[modelM][];

            for (int i = 0; i < modelM; i++) {
                labels[i] = new double[modelN];
                for (int j = 0; j < modelN; j++) {
                    labels[i][j] = Linear.patentPredictValue(models[i][j], nodes);
                    updateMinMaxModuleRawLabel(labels[i][j]);
                }
            }
            minMaxModuleRawLabels.add(labels);
            predictedLabel = MyUtil.MinMaxModule(labels, 0, threshold);
            Linear.printf(out, "final %g\n", predictedLabel);
            if (predictedLabel == realLabel)
                ++correct;
            error += (predictedLabel-realLabel)*(predictedLabel-realLabel);
            sump += predictedLabel;
            sumr += realLabel;
            sumpp += predictedLabel*predictedLabel;
            sumrr += realLabel*realLabel;
            sumpr += predictedLabel*realLabel;
            ++total;
        }
        if (models[0][0].solverType.isSupportVectorRegression()) {
            Linear.info("Mean Squared error = %g (regression)%n", error/total);
            Linear.info("Squared correlation coefficient = %g (regression)%n", ((total*sumpr-sump*sumr)*(total*sumpr-sump*sumr))/((total*sumpp-sump*sump)*(total*sumrr-sumr*sumr)));
        } else {
            Linear.info("Accuracy = %g%% (%d/%d)%n", (double)correct/total*100, correct, total);
        }
    }

    static void minMaxModulePredict(Model[][] models, String filename) throws IOException {
        Linear.info("Min_Max_Module Predict Begin...\n");
        long start = System.currentTimeMillis();
        Writer writer;
        writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), Linear.FILE_CHARSET));
        minMaxModuleRawLabels = new ArrayList<>();
        minRawLabel = 0;
        maxRawLabel = 0;
        doMinMaxModulePredict(models, writer);
        Linear.closeQuietly(writer);
        long end = System.currentTimeMillis();
        Linear.info("Min_Max_Module Predict End, Time Used: %dms\n\n\n", end-start);
        evaluation.initMinMaxModule(minMaxModuleRawLabels, filename, minRawLabel, maxRawLabel, threshold);
        evaluation.minMaxModuleEvaluation();
    }

    static void basicPredict(Model model, String filename) throws IOException {
        Linear.info("Basic Predict Begin...\n");
        long start = System.currentTimeMillis();
        Writer writer;
        writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), Linear.FILE_CHARSET));
        basicRawLabels = new ArrayList<>();
        minRawLabel = 0;
        maxRawLabel = 0;
        doBasicPredict(model, writer);
        Linear.closeQuietly(writer);
        long end = System.currentTimeMillis();
        Linear.info("Basic Predict End, Time Used: %dms\n\n\n", end-start);
        evaluation.initBasic(basicRawLabels, filename, minRawLabel, maxRawLabel, threshold);
        evaluation.basicEvaluation();
    }
}
