package hex.tree.xgboost.util;

public class FeatureScore {
    
    public static final String GAIN_KEY = "gain";
    public static final String COVER_KEY = "cover";
    
    public int _frequency;
    public float _gain;
    public float _cover;

    public void add(FeatureScore fs) {
        _frequency += fs._frequency;
        _gain += fs._gain;
        _cover += fs._cover;
    }
}
