import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by DELL on 2017/3/26.
 */
public class MyUtil {

    static class Record {
        Double label;
        Feature[] features;

        Record(double l, Feature[] f) {
            label = l;
            features = f;
        }
    }

    static double transferLabel(String token) throws NumberFormatException {
        if(token.charAt(0) == 'A')
            return 1;
        if(token.charAt(0) == 'B' || token.charAt(0) == 'C' || token.charAt(0) == 'D')
            return -1;
        throw new NumberFormatException();
    }

    static double MinMaxModule(double[][] labels, double standard, double threshold) {
        boolean notBreak;
        boolean allZero = true;
        for(double[] labelList : labels) {
            notBreak = true;
            for(double label : labelList){
                if(label < standard - threshold) {
                    notBreak = false;
                    allZero = false;
                    break;
                }
                if(label > standard + threshold) {
                    allZero = false;
                }
            }
            if(notBreak && !allZero)
                return 1;
        }
        if(allZero)
            return 0;
        return -1;
    }

    static void multithreadingJoin(Thread[][] threads) {
        for(Thread[] threadGroup : threads) {
            for(Thread thread : threadGroup) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    continue;
                }
            }
        }
    }

    static int[] listToArray(ArrayList<Integer> list) {
        int[] res = new int[list.size()];
        Iterator<Integer> integerIterator = list.iterator();
        int i = 0;
        while (integerIterator.hasNext())
            res[i++] = integerIterator.next();
        return res;
    }
}
