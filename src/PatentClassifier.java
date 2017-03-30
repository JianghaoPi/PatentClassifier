import java.io.IOException;

/**
 * Created by DELL on 2017/3/29.
 */
public class PatentClassifier {
    private static final String trainFile = "../data/train.txt";
    private static final String testFile = "../data/test.txt";
    private static final String basicResult = "../file/basic_result";
    private static final String randomMinMaxModuleResult = "../file/ramdom_min_max_module_result";
    private static final String labeledMinMaxModuleResult = "../file/labeled_min_max_module_result";

    private static void Basic() throws IOException, InvalidInputDataException {
        Model model = MyTrain.basicTrain();
        if (model == null)
            return;
        MyPredict.basicPredict(model, basicResult);
    }

    private static void RandomMinMaxModule() throws IOException, InvalidInputDataException {
        Model[][] models = MyTrain.randomMinMaxModuleTrain();
        if (models[0][0] == null)
            return;
        MyPredict.minMaxModulePredict(models, randomMinMaxModuleResult);
    }

    private static void LabeledMinMaxModule() throws IOException, InvalidInputDataException {
        Model[][] models = MyTrain.labeledMinMaxModuleTrain();
        if (models[0][0] == null)
            return;
        MyPredict.minMaxModulePredict(models, labeledMinMaxModuleResult);
    }

    public static void main(String[] args) throws Exception {
        new MyTrain(trainFile);
        int dimensions = MyTrain.getDimensions();
        double bias = MyTrain.getBias();
        System.out.printf("Dimensions: %d \t Bias: %g\n", dimensions, bias);
        new MyPredict(testFile, dimensions, bias);
        Basic();
        RandomMinMaxModule();
        LabeledMinMaxModule();
    }
}
