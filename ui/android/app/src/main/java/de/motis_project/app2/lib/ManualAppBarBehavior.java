package de.motis_project.app2.lib;

import android.content.Context;
import android.support.design.widget.AppBarLayout;
import android.support.design.widget.CoordinatorLayout;
import android.util.AttributeSet;
import android.view.View;

/**
 * AppBars with this behavior do not automatically expand/collapse when scrolling
 * other views. They can only be expanded/collapsed using
 * {@link AppBarLayout#setExpanded(boolean, boolean)}.
 */
public class ManualAppBarBehavior extends AppBarLayout.Behavior {
    public ManualAppBarBehavior() {
    }

    public ManualAppBarBehavior(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @Override
    public boolean onStartNestedScroll(CoordinatorLayout parent, AppBarLayout child,
                                       View directTargetChild, View target,
                                       int nestedScrollAxes, int type) {
        return false;
    }
}
