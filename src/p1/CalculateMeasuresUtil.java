package p1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

  // calculate median

  private static double calculateMedian(List<Double> doubleList) {
    return 0;
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
