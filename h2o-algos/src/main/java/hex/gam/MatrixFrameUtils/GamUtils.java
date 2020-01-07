package hex.gam.MatrixFrameUtils;

import hex.Model;
import hex.gam.GAMModel.GAMParameters;
import hex.glm.GLMModel.GLMParameters;
import water.MemoryManager;
import water.fvec.Frame;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

public class GamUtils {
  /***
   * Allocate 3D array to store various info.
   * @param num2DArrays
   * @param parms
   * @param fileMode: 0: allocate for transpose(Z), 1: allocate for S, 2: allocate for t(Z)*S*Z
   * @return
   */
  public static double[][][] allocate3DArray(int num2DArrays, GAMParameters parms, int fileMode) {
    double[][][] array3D = new double[num2DArrays][][];
    for (int frameIdx = 0; frameIdx < num2DArrays; frameIdx++) {
      int numKnots = parms._k[frameIdx];
      switch (fileMode) {
        case 0: array3D[frameIdx] = MemoryManager.malloc8d(numKnots-1, numKnots); break;
        case 1: array3D[frameIdx] = MemoryManager.malloc8d(numKnots, numKnots); break;
        case 2: array3D[frameIdx] = MemoryManager.malloc8d(numKnots-1, numKnots-1); break;
        default: throw new IllegalArgumentException("fileMode can only be 0, 1 or 2.");
      }
    }
    return array3D;
  }

  public static void copy2DArray(double[][] src_array, double[][] dest_array) {
    int numRows = src_array.length;
    for (int colIdx = 0; colIdx < numRows; colIdx++) { // save zMatrix for debugging purposes or later scoring on training dataset
      System.arraycopy(src_array[colIdx], 0, dest_array[colIdx], 0,
              src_array[colIdx].length);
    }
  }

  /***
   * from Stack overflow by dfa/user1079877
   * @param fields
   * @param type
   * @return
   */
  public static List<Field> getAllFields(List<Field> fields, Class<?> type) {
    fields.addAll(Arrays.asList(type.getDeclaredFields()));

    if (type.getSuperclass() != null) {
      getAllFields(fields, type.getSuperclass());
    }

    return fields;
  }

  public static GLMParameters copyGAMParams2GLMParams(GAMParameters parms, Frame trainData) {
    GLMParameters glmParam = new GLMParameters();
    Field[] field1 = GAMParameters.class.getDeclaredFields();
    setParamField(parms, glmParam, false, field1);
    Field[] field2 = Model.Parameters.class.getDeclaredFields();
    setParamField(parms, glmParam, true, field2);
    glmParam._train = trainData._key;
    return glmParam;
  }
  
  public static void setParamField(GAMParameters parms, GLMParameters glmParam, boolean superClassParams, Field[] gamFields) {
    // assign relevant GAMParameter fields to GLMParameter fields
    List<String> gamOnlyList = Arrays.asList(new String[]{"_k", "_gam_X", "_bs", "_scale", "_train", "_saveZMatrix", 
            "_saveGamCols", "_savePenaltyMat", "_ignored_columns"});
    for (Field oneField : gamFields) {
      if (oneField.getName().equals("_repsonse_column"))
        System.out.println("whoe");
      try {
        if (!gamOnlyList.contains(oneField.getName())) {
          Field glmField = superClassParams?glmParam.getClass().getSuperclass().getDeclaredField(oneField.getName())
                  :glmParam.getClass().getDeclaredField(oneField.getName());
          glmField.set(glmParam, oneField.get(parms));
        }
      } catch (IllegalAccessException e) { // suppress error printing
        ;
      } catch (NoSuchFieldException e) {
        ;
      }
    }
  }

  public static int locateBin(double xval, double[] knots) {
    if (xval <= knots[0])  //small short cut
      return 0;
    int highIndex = knots.length-1;
    if (xval >= knots[highIndex]) // small short cut
      return (highIndex-1);

    int binIndex = 0;
    int count = 0;
    int numBins = knots.length;
    int lowIndex = 0;

    while (count < numBins) {
      int tryBin = (int) Math.floor((highIndex+lowIndex)*0.5);
      if ((xval >= knots[tryBin]) && (xval < knots[tryBin+1]))
        return tryBin;
      else if (xval > knots[tryBin])
        lowIndex = tryBin;
      else if (xval < knots[tryBin])
        highIndex = tryBin;

      count++;
    }
    return binIndex;
  }
}
