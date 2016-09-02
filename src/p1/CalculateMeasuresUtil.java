package p1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static java.util.Collections.swap;

public final class CalculateMeasuresUtil {

  // calculate average

  private static double calculateAverage(List<Double> doubleList) {
    double sum = 0.0;
    for (Double d : doubleList) {
      sum += d;
    }
    return sum / doubleList.size();
  }

  // calculate mode

  private static Map<Double, Integer> getCountMap(List<Double> motherAgeList) {
    Map<Double, Integer> countMap = new HashMap<>();
    for (Double d : motherAgeList) {
      Integer count = countMap.get(d);
      if (count == null) {
        count = 1;
      } else {
        count++;
      }
      countMap.put(d, count);
    }
    return countMap;
  }

  private static double getMaxCountValue(Map<Double, Integer> countMap) {
    int maxCount = 0;
    double maxCountValue = Double.NaN;
    for (Map.Entry<Double, Integer> entry : countMap.entrySet()) {
      Integer count = entry.getValue();
      if (count > maxCount) {
        maxCount = count;
        maxCountValue = entry.getKey();
      }
    }
    return maxCountValue;
  }

  private static double calculateMode(List<Double> motherAgeList) {
    Map<Double, Integer> countMap = getCountMap(motherAgeList);
    return getMaxCountValue(countMap);
  }

  // calculate median (quick select algorithm)

  private static int getPivotIndex(int left, int right) {
    // fixed seed for reproducible results
    Random random = new Random(0L);
    return left + random.nextInt(right - left + 1);
  }

  /**
   * Partition the list into two parts: smaller than the pivot value and larger than the pivot value
   * and return the new value of the pivot index
   *
   * @param right index (inclusive)
   * @return new value of the pivot index
   */
  private static int partition(List<Double> doubleList, int left, int right, int pivotIndex) {
    double pivotValue = doubleList.get(pivotIndex);
    swap(doubleList, pivotIndex, right);
    int resultIndex = left;
    for (int i = left; i < right; i++) {
      if (doubleList.get(i) < pivotValue) {
        swap(doubleList, resultIndex, i);
        resultIndex++;
      }
    }
    swap(doubleList, right, resultIndex);
    return resultIndex;
  }

  /**
   * Find the nth smallest element
   *
   * @param right index (inclusive)
   * @param n nth smallest element
   * @return the value of the nth smallest element
   */
  private static double select(List<Double> doubleList, int left, int right, int n) {
    double resultElement;
    if (left == right) {
      // just one element
      resultElement = doubleList.get(left);
    } else {
      int pivotIndex = getPivotIndex(left, right);
      pivotIndex = partition(doubleList, left, right, pivotIndex);
      if (n == pivotIndex) {
        resultElement = doubleList.get(n);
      } else if (n < pivotIndex) {
        // search in the lower (left) part
        resultElement = select(doubleList, left, pivotIndex - 1, n);
      } else {
        // search in the upper (right) part
        resultElement = select(doubleList, pivotIndex + 1, right, n);
      }
    }
    return resultElement;
  }

  private static double select(List<Double> doubleList, int n) {
    // make a copy to preserve the original because partition() rearranges items in place
    List<Double> doubleListCopy = new ArrayList<>(doubleList);
    return select(doubleListCopy, 0, doubleList.size() - 1, n);
  }

  private static double calculateMedian(List<Double> doubleList) {
    double median;
    int size = doubleList.size();
    if (size % 2 == 0) {
      // even
      int midN = size / 2;
      double m1 = select(doubleList, midN - 1);
      double m2 = select(doubleList, midN);
      median = (m1 + m2) / 2.0;
    } else {
      // odd
      int midN = size / 2;
      median = select(doubleList, midN);
    }
    return median;
  }

  // filter by infant weight, get mother age list

  private static List<Double> getMotherAgeList(
      double[][] birthWeightTable, int numberOfRows, double minInfantWeight, double maxInfantWeight) {
    List<Double> motherAgeList = new ArrayList<>();
    int size = Math.min(birthWeightTable.length, numberOfRows);
    for (int i = 0; i < size; i++) {
      double[] row = birthWeightTable[i];
      double motherAge = row[0];
      double infantWeight = row[1];
      if (minInfantWeight <= infantWeight && infantWeight <= maxInfantWeight) {
        motherAgeList.add(motherAge);
      }
    }
    return motherAgeList;
  }

  // calculate measures

  public static Measures calculateMeasures(
      double[][] birthWeightTable, int numberOfRows, double minInfantWeight, double maxInfantWeight) {
    double average;
    double median;
    double mode;
    List<Double> motherAgeList = getMotherAgeList(birthWeightTable, numberOfRows, minInfantWeight, maxInfantWeight);
    if (motherAgeList.isEmpty()) {
      average = Double.NaN;
      median = Double.NaN;
      mode = Double.NaN;
    } else {
      average = calculateAverage(motherAgeList);
      median = calculateMedian(motherAgeList);
      mode = calculateMode(motherAgeList);
    }
    return new Measures(average, median, mode);
  }

  public static final class Measures {

    public final double average;
    public final double median;
    public final double mode;

    private Measures(double average, double median, double mode) {
      this.average = average;
      this.median = median;
      this.mode = mode;
    }

  }

  private CalculateMeasuresUtil() {
  }

}
