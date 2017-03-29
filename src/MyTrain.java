import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by DELL on 2017/3/25.
 */
public class MyTrain {
    private static final int labelTypeNum = 4;
    private static final char labelBegin = 'A';
    private static int subPosSetNum;
    private static double bias = 1;
    private static boolean cross_validation = false;
    private static int nr_fold;
    private static Parameter parameter = null;
    private static Problem problem = null;
    private static ArrayList<MyUtil.Record>[] trainSet;
    private static int dimensions;

    MyTrain(String inputFileName) throws IOException, InvalidInputDataException {
        trainSet = new ArrayList[labelTypeNum];
        initialize_parameters();
        readDataFile(new File(inputFileName));
    }

    private static void initialize_parameters() {
        parameter = new Parameter();
        parameter.setSolverType(SolverType.L2R_L2LOSS_SVC_DUAL);
        parameter.setC(1);
        parameter.setMaxIters(1000);
        parameter.setEps(0.00001);
        bias = -1;
        cross_validation = false;
        nr_fold = 10;
        subPosSetNum = 6;
    }

    static int getDimensions() {
        return dimensions;
    }

    static double getBias() {
        return bias;
    }

    private static void readDataFile(File file) throws IOException, InvalidInputDataException {
        dimensions = 0;
        int lineNr = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(file))){
            while(true) {
                String line = br.readLine();
                if(line == null)
                    break;
                lineNr++;

                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                String token;
                try {
                    token = st.nextToken();
                }catch (NoSuchElementException e) {
                    throw new InvalidInputDataException("empty line", file, lineNr, e);
                }

                int setSequence = token.charAt(0) - labelBegin;

                if (trainSet[setSequence] == null) {
                    trainSet[setSequence] = new ArrayList<>();
                }

                double label;
                try {
                    label = MyUtil.transferLabel(token);
                } catch (NumberFormatException e) {
                    throw new InvalidInputDataException("invalid label: " + token, file, lineNr, e);
                }

                int m = st.countTokens() / 2;
                Feature[] features;
                if (bias >= 0) {
                    features = new Feature[m+1];
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
                if (m > 0) {
                    dimensions = Math.max(dimensions, features[m-1].getIndex());
                }
                trainSet[setSequence].add(new MyUtil.Record(label, features));
            }
            Linear.info("Train Set Size: %d\n", lineNr);
        }
    }

    private static void crossValidate() {
        double totalError = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[problem.l];

        long start, stop;
        start = System.currentTimeMillis();
        Linear.crossValidation(problem, parameter, nr_fold, target);
        stop = System.currentTimeMillis();
        System.out.println("time: " + (stop - start) + "ms");

        if (parameter.solverType.isSupportVectorRegression()) {
            for (int i = 0; i < problem.l; i++) {
                double y = problem.y[i];
                double v = target[i];
                totalError += (v - y) * (v - y);
                sumv += v;
                sumy += y;
                sumvv += v * v;
                sumyy += y * y;
                sumvy += v * y;
            }
            System.out.printf("Cross Validation Mean_squared_error = %g%n", totalError/problem.l);
            System.out.printf("Cross Validation Squared_correlation_corfficient = %g%n", ((problem.l*sumvy-sumv*sumy)*(problem.l*sumvy-sumv*sumy))/((problem.l*sumvv-sumv*sumv)*(problem.l*sumyy-sumy*sumy)));
        } else {
            int totalCorrect = 0;
            for (int i = 0; i < problem.l; i++)
                if (target[i] == problem.y[i])
                    ++totalCorrect;
            System.out.printf("correct: %d%n", totalCorrect);
            System.out.printf("Cross Validation Accuracy = %g%%%n", 100.0*totalCorrect/problem.l);
        }
    }

    private static Problem basicProblem() {
        ArrayList<MyUtil.Record> data = new ArrayList<>();
        for (int i = 0; i < labelTypeNum; i++) {
            data.addAll(trainSet[i]);
        }
        Collections.shuffle(data);
        return constructProblem(data, dimensions, bias);
    }

    private static Problem constructProblem(List<MyUtil.Record> data, int offset, int length, int dimensions, double bias) {
        Problem problem = new Problem();
        problem.bias = bias;
        problem.l = Math.min(data.size() - offset, length);
        problem.n = dimensions;
        if (bias >= 0)
            problem.n++;
        problem.x = new Feature[problem.l][];
        for (int i = 0; i < problem.l; i++) {
            problem.x[i] = data.get(offset + i).features;
            if (bias >= 0) {
                assert problem.x[i][problem.x[i].length-1] == null;
                problem.x[i][problem.x[i].length-1] = new FeatureNode(dimensions+1, bias);
            }
        }

        problem.y = new double[problem.l];
        for (int i = 0; i < problem.l; i++)
            problem.y[i] = data.get(offset+i).label;
        return problem;
    }

    private static Problem constructProblem(List<MyUtil.Record> data, int dimensions, double bias) {
        return constructProblem(data, 0, data.size(), dimensions, bias);
    }

    private static int randomCalNegSetNum(int posSize, int negSize) {
        int subSetSize = posSize/subPosSetNum;
        return negSize/subSetSize;
    }

    private static void randomTrainSub(Model[][] res, int i, int j, ArrayList<MyUtil.Record> pos, ArrayList<MyUtil.Record> neg, int posOffset, int negOffset, int posLength, int negLength) {
        ArrayList<MyUtil.Record> set = new ArrayList<>();
        set.addAll(pos.subList(posOffset, posOffset+posLength-1));
        set.addAll(neg.subList(negOffset, negOffset+negLength-1));
        Collections.shuffle(set);
        res[i][j] = Linear.train(constructProblem(set, dimensions, bias), parameter);
    }

    private static int[][] randomConstructInfo(int posSize, int negSize, int constPosSetNum, int constNegSetNum) {
        Linear.info("Positive Subset Number: %d\n", constPosSetNum);
        Linear.info("Negative Subset Number: %d\n", constNegSetNum);

        int posOffset = 0, negOffset = 0;
        int posLength, negLength;
        int posSetNum = constPosSetNum, negSetNum =constNegSetNum;
        int[] posInfoOffset = new int[constPosSetNum];
        int[] posInfoLength = new int[constPosSetNum];
        int[] negInfoOffset = new int[constNegSetNum];
        int[] negInfoLength = new int[constNegSetNum];

        for (int i = 0; i < constPosSetNum; i++) {
            posInfoOffset[i] = posOffset;
            posLength = posSize/posSetNum;
            posInfoLength[i] = posLength;
            posSize -= posLength;
            posSetNum--;
            posOffset += posLength;
        }
        for (int i = 0; i < constNegSetNum; i++) {
            negInfoOffset[i] = negOffset;
            negLength = negSize/negSetNum;
            negInfoLength[i] = negLength;
            negSize -= negLength;
            negSetNum--;
            negOffset += negLength;
        }
        return new int[][]{posInfoOffset, posInfoLength, negInfoOffset, negInfoLength};
    }

    private static int[][] labeledConstructInfo() {
        int posSize = trainSet[0].size();
        int standardPosLength = posSize/subPosSetNum;
        int posOffset = 0, negOffset = 0;
        int posLength, negLength;
        int posSetNum = subPosSetNum;
        int[] posInfoOffset = new int[subPosSetNum];
        int[] posInfoLength = new int[subPosSetNum];

        for (int i = 0; i < subPosSetNum; i++) {
            posInfoOffset[i] = posOffset;
            posLength = posSize/posSetNum;
            posInfoLength[i] = posLength;
            posSize -= posLength;
            posOffset += posLength;
            posSetNum--;
        }

        int negOffsetBase = 0;
        ArrayList<Integer> negInfoOffsetList = new ArrayList<>();
        ArrayList<Integer> negInfoLengthList = new ArrayList<>();
        for (int i = 1; i < trainSet.length; i++) {
            int constNegSize = trainSet[i].size();
            int negSize = constNegSize;
            int tmp = constNegSize/standardPosLength;
            int negSetNum = (tmp == 0) ? 1 : tmp;
            while (negOffset - negOffsetBase < constNegSize) {
                negLength = negSize/negSetNum;
                negInfoOffsetList.add(negOffset);
                negInfoLengthList.add(negLength);
                negSize -= negLength;
                negOffset += negLength;
                negSetNum--;
            }
            negOffsetBase += constNegSize;
        }
        return new int[][]{posInfoOffset, posInfoLength, MyUtil.listToArray(negInfoOffsetList), MyUtil.listToArray(negInfoLengthList)};
    }

    private static ArrayList<MyUtil.Record>[] constructArrayList() {
        ArrayList<MyUtil.Record>[] res = new ArrayList[2];
        res[0] = new ArrayList<>();
        res[1] = new ArrayList<>();
        res[0].addAll(trainSet[0]);
        for (int i = 1; i<labelTypeNum; i++)
            res[1].addAll(trainSet[i]);
        return res;
    }

    static Model basicTrain() throws IOException, InvalidInputDataException {
        Linear.info("Basic Train Begin...\n");
        long start = System.currentTimeMillis();
        problem = basicProblem();
        if (cross_validation) {
            crossValidate();
            return null;
        } else {
            Model model = Linear.train(problem, parameter);
            long end = System.currentTimeMillis();
            Linear.info("Basic Train End, Time Used: %dms\n\n\n", end - start);
            return model;
        }
    }

    static Model[][] randomMinMaxModuleTrain() throws IOException, InvalidInputDataException {
        Linear.info("Random Min_Max_Module Train Begin...\n");
        long start = System.currentTimeMillis();
        ArrayList<MyUtil.Record>[] arrayLists = constructArrayList();
        ArrayList<MyUtil.Record> pos = arrayLists[0];
        ArrayList<MyUtil.Record> neg = arrayLists[1];
        Collections.shuffle(pos);
        Collections.shuffle(neg);
        int posSize = pos.size();
        int negSize = neg.size();
        int constNegSetNum = randomCalNegSetNum(posSize, negSize);
        int[][] info = randomConstructInfo(posSize, negSize, subPosSetNum, constNegSetNum);
        int[] posInfoOffset = info[0];
        int[] posInfoLength = info[1];
        int[] negInfoOffset = info[2];
        int[] negInfoLength = info[3];
        Model[][] res = new Model[subPosSetNum][];
        TrainSubThread[][] threads = new TrainSubThread[subPosSetNum][];
        for (int i = 0; i < subPosSetNum; i++) {
            res[i] = new Model[constNegSetNum];
            threads[i] = new TrainSubThread[constNegSetNum];
            for (int j = 0; j < constNegSetNum; j++) {
                Linear.info("Train Subset i: %d size: %d \t j: %d size: %d\n", i, posInfoLength[i], j, posInfoLength[j]);
                threads[i][j] = new TrainSubThread(res, i, j, pos, neg, posInfoOffset[i], negInfoOffset[j], posInfoLength[i], negInfoLength[j]);
                threads[i][j].start();
            }
        }
        MyUtil.multithreadingJoin(threads);
        long end = System.currentTimeMillis();
        Linear.info("Random Min_Max_Module Train End, Time Used: %dms\n\n\n", end - start);
        return res;
    }

    static Model[][] labeledMinMaxModuleTrain() throws IOException, InvalidInputDataException {
        Linear.info("Labeled Min_Max_Module Train Begin...\n");
        long start = System.currentTimeMillis();
        ArrayList<MyUtil.Record>[] arrayLists = constructArrayList();
        ArrayList<MyUtil.Record> pos = arrayLists[0];
        ArrayList<MyUtil.Record> neg = arrayLists[1];
        int[][] info = labeledConstructInfo();
        int[] posInfoOffset = info[0];
        int[] posInfoLength = info[1];
        int[] negInfoOffset = info[2];
        int[] negInfoLength = info[3];
        int constNegSetNum = negInfoOffset.length;
        Model[][] res = new Model[subPosSetNum][];
        TrainSubThread[][] threads = new TrainSubThread[subPosSetNum][];
        for (int i = 0; i < subPosSetNum; i++) {
            res[i] = new Model[constNegSetNum];
            threads[i] = new TrainSubThread[constNegSetNum];
            for (int j = 0; j < constNegSetNum; j++) {
                Linear.info("Train SubSet i: %d size: %d \t j: %d size: %d\n", i, posInfoLength[i], j, posInfoLength[j]);
                threads[i][j] = new TrainSubThread(res, i, j, pos, neg, posInfoOffset[i], negInfoOffset[j], posInfoLength[i], negInfoLength[j]);
                threads[i][j].start();
            }
        }
        MyUtil.multithreadingJoin(threads);
        long end = System.currentTimeMillis();
        Linear.info("Labeled Min_Max_Module Train End, Time Used: %dms\n\n\n", end - start);
        return res;
    }

    private static class TrainSubThread extends Thread {
        private Model[][] res;
        private int i;
        private int j;
        private ArrayList<MyUtil.Record> pos;
        private ArrayList<MyUtil.Record> neg;
        private int posOffset;
        private int negOffset;
        private int posLength;
        private int negLength;

        TrainSubThread(Model[][] res, int i, int j, ArrayList<MyUtil.Record> pos, ArrayList<MyUtil.Record> neg, int posOffset, int negOffset, int posLength, int negLength) {
            this.res = res;
            this.i = i;
            this.j = j;
            this.pos = pos;
            this.neg = neg;
            this.posOffset = posOffset;
            this.negOffset = negOffset;
            this.posLength = posLength;
            this.negLength = negLength;
        }

        @Override
        public void run() {
            ArrayList<MyUtil.Record> set = new ArrayList<>();
            set.addAll(pos.subList(posOffset, posOffset+posLength-1));
            set.addAll(neg.subList(negOffset, negOffset+negLength-1));
            Collections.shuffle(set);
            res[i][j] = Linear.train(constructProblem(set, dimensions, bias), parameter);
        }
    }
}
