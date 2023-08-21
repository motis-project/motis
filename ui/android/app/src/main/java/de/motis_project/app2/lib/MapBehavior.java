package de.motis_project.app2.lib;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;

import androidx.coordinatorlayout.widget.CoordinatorLayout;
import androidx.core.view.ViewCompat;


public class MapBehavior<V extends View> extends CoordinatorLayout.Behavior<V> {
    private static final String TAG = "MapBehavior";

    private View topView;
    private View bottomView;

    private int height = 0;
    private int top = 0;

    public MapBehavior() {
        super();
    }

    public MapBehavior(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public View getTopView() {
        return topView;
    }

    public void setTopView(View topView) {
        this.topView = topView;
    }

    public View getBottomView() {
        return bottomView;
    }

    public void setBottomView(View bottomView) {
        this.bottomView = bottomView;
    }

    private void calculateHeight(CoordinatorLayout parent) {
        int parentHeight = parent.getHeight();
        int topHeight = (topView == null || topView.getVisibility() == View.GONE)
                ? 0 : (topView.getHeight() + topView.getTop());
        int bottomHeight = (bottomView == null || bottomView.getVisibility() == View.GONE)
                ? 0 : (parentHeight - bottomView.getTop());
        top = topHeight;
        height = Math.max(0, parentHeight - topHeight - bottomHeight);
    }

    @Override
    public boolean layoutDependsOn(CoordinatorLayout parent, V child, View dependency) {
        if (dependency == topView || dependency == bottomView) {
            return true;
        } else {
            return super.layoutDependsOn(parent, child, dependency);
        }
    }

    @Override
    public boolean onDependentViewChanged(CoordinatorLayout parent, V child, View dependency) {
        int oldHeight = child.getHeight();
        int oldTop = child.getTop();
        calculateHeight(parent);
        if (height != oldHeight || top != oldTop) {
            ViewGroup.LayoutParams layoutParams = child.getLayoutParams();
            layoutParams.height = height;
            child.setLayoutParams(layoutParams);
            ViewCompat.offsetTopAndBottom(child, top - child.getTop());
            return true;
        } else {
            return false;
        }
    }

    @Override
    public boolean onMeasureChild(CoordinatorLayout parent, V child, int parentWidthMeasureSpec,
                                  int widthUsed, int parentHeightMeasureSpec, int heightUsed) {
        if (topView == null && bottomView == null) {
            return false;
        } else {
            calculateHeight(parent);
            parent.onMeasureChild(child, parentWidthMeasureSpec, widthUsed,
                    View.MeasureSpec.makeMeasureSpec(height, View.MeasureSpec.AT_MOST), heightUsed);
            return true;
        }
    }

    @Override
    public boolean onLayoutChild(CoordinatorLayout parent, V child, int layoutDirection) {
        if (topView == null && bottomView == null) {
            return false;
        } else {
            parent.onLayoutChild(child, layoutDirection);
            calculateHeight(parent);
            ViewCompat.offsetTopAndBottom(child, top - child.getTop());
            return true;
        }
    }

    /**
     * A utility function to get the {@link MapBehavior} associated with the {@code view}.
     *
     * @param view The {@link View} with {@link MapBehavior}.
     * @return The {@link MapBehavior} associated with the {@code view}.
     */
    @SuppressWarnings("unchecked")
    public static <V extends View> MapBehavior<V> from(V view) {
        ViewGroup.LayoutParams params = view.getLayoutParams();
        if (!(params instanceof CoordinatorLayout.LayoutParams)) {
            throw new IllegalArgumentException("The view is not a child of CoordinatorLayout");
        }
        CoordinatorLayout.Behavior behavior = ((CoordinatorLayout.LayoutParams) params)
                .getBehavior();
        if (!(behavior instanceof MapBehavior)) {
            throw new IllegalArgumentException(
                    "The view is not associated with MapBehavior");
        }
        return (MapBehavior<V>) behavior;
    }
}
