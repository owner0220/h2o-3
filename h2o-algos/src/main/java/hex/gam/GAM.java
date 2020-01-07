package hex.gam;

import hex.ModelBuilder;
import hex.ModelCategory;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.Link;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;

import java.util.Arrays;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      if (!_parms._family.equals(Family.gaussian))
        error("_family", "Only gaussian family is supported for now.");
      if (!_parms._link.equals(Link.identity) && !_parms._link.equals(Link.family_default))
        error("_link", "Only identity or family_default link is supported for now.");
      if (_parms._gam_X == null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k == null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10 ? (int) _train.numRows() : 10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      }
      if ((_parms._saveGamCols || _parms._saveZMatrix) && ((_train.numCols() - 1 + _parms._k.length) < 2))
        error("_saveGamCols/_saveZMatrix", "can only be enabled if we number of predictors plus" +
                " Gam columns in _gamX exceeds 2");
    }
  }

  @Override
  public void checkDistributions() {  // will be called in ModelBuilder.java
    if (!_response.isNumeric()) {
      error("_response", "Expected a numerical response, but instead got response with " + _response.cardinality() + " categories.");
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return false; // no use, we don't output probabilities
  }

  @Override
  protected int init_getNClass() {
    return 1; // only regression is supported for now
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override
  protected void checkMemoryFootPrint_impl() {
    ;
  }

  private class GAMDriver extends Driver {
    boolean _centerGAM = false; // true if we need to constraint GAM columns
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat;  // store for each GAM predictir the penalty matrix
    String[][] _gamColnames;  // store column names of GAM columns

    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * @return
     */
    Frame adaptTrain() {
      Frame orig = _parms.train();  // contain all columns, _train contains only predictors and responses
      Scope.track(orig);
      int numGamFrame = _parms._gam_X.length;
      Key<Frame>[] gamFramesKey = new Key[numGamFrame];  // store the Frame keys of generated GAM column
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      _centerGAM = (numGamFrame > 1) || (_train.numCols() - 1 + numGamFrame) >= 2; // only need to center when predictors > 1
      _zTranspose = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 0) : null;
      _penalty_mat = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 2) : GamUtils.allocate3DArray(numGamFrame, _parms, 1);
      for (int index = 0; index < numGamFrame; index++) {
        final Frame predictVec = new Frame(new String[]{_parms._gam_X[index]}, new Vec[]{orig.vec(_parms._gam_X[index])});  // extract the vector to work on
        final int numKnots = _parms._k[index];  // grab number of knots to generate
        final double scale = _parms._scale == null ? 1.0 : _parms._scale[index];
        final BSType splineType = _parms._bs[index];
        final int frameIndex = index;
        final String[] newColNames = new String[numKnots];
        for (int colIndex = 0; colIndex < numKnots; colIndex++) {
          newColNames[colIndex] = _parms._gam_X[index] + "_" + splineType.toString() + "_" + colIndex;
        }
        generateGamColumn[frameIndex] = new RecursiveAction() {
          @Override
          protected void compute() {
            GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots, null, predictVec,
                    _parms._standardize, _centerGAM, scale).doAll(numKnots, Vec.T_NUM, predictVec);
            Frame oneAugmentedColumn = genOneGamCol.outputFrame(Key.make(), newColNames,
                    null);
            if (_centerGAM) {  // calculate z transpose
              oneAugmentedColumn = genOneGamCol.de_centralize_frame(oneAugmentedColumn,
                      predictVec.name(0) + "_" + splineType.toString() + "_decenter_", _parms);
              GamUtils.copy2DArray(genOneGamCol._ZTransp, _zTranspose[frameIndex]); // copy transpose(Z)
              double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
                      genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
              GamUtils.copy2DArray(transformedPenalty, _penalty_mat[frameIndex]);
            } else
              GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat[frameIndex]); // copy penalty matrix
            gamFramesKey[frameIndex] = oneAugmentedColumn._key;
            DKV.put(oneAugmentedColumn);
          }
        };
      }
      ForkJoinTask.invokeAll(generateGamColumn);
      _gamColnames = new String[numGamFrame][];
      for (int frameInd = 0; frameInd < numGamFrame; frameInd++) {  // append the augmented columns to _train
        Frame gamFrame = gamFramesKey[frameInd].get();
        _train.add(gamFrame);
        _gamColnames[frameInd] = new String[gamFrame.names().length];
        System.arraycopy(gamFrame.names(), 0, _gamColnames[frameInd], 0, gamFrame.names().length);
        Scope.track(gamFrame);
      }
      return _train;
    }

    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      _job.update(0, "Initializing model training");

      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = null;
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        Frame newTFrame = new Frame(adaptTrain());  // get frames with correct predictors and spline functions
        DKV.put(newTFrame);

        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, newTFrame));
        model.delete_and_lock(_job);
        model._centerGAM = _centerGAM;
        if (_parms._saveGamCols)
          model._output._gamTransformedTrain = newTFrame._key;  // access after model is built

        _job.update(0, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame); // obtained GLM model
        Scope.track_generic(glmModel);
        // convert to GAM model
        // call adaptTeatForTrain() to massage frame before scoring.
        // call model.makeModelMetrics
        // create model summary by calling createModelSummaryTable or something like that
        model.update(_job);

        // build GAM Model Metrics
      } finally {
        if (model != null) 
          model.unlock(_job);
      }
    }
    
    GLMModel buildGLMModel(GAMParameters parms, Frame trainData) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData);  // copy parameter from GAM to GLM
      GLMModel model = new GLM(glmParam, _penalty_mat, _gamColnames).trainModel().get();
      return model;
    }
  }
}
