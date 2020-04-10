package de.motis_project.app2.ppr;

import android.content.Context;
import android.support.annotation.Nullable;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.AttributeSet;

import de.motis_project.app2.lib.SimpleDividerItemDecoration;


public class RouteStepsView extends RecyclerView {
    private static final String TAG = "RouteStepsView";

    private final LinearLayoutManager layoutManager = new LinearLayoutManager(getContext());

    public RouteStepsView(Context context) {
        super(context);
        init();
    }

    public RouteStepsView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public RouteStepsView(Context context, @Nullable AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init();
    }

    private void init() {
        addItemDecoration(new SimpleDividerItemDecoration(getContext()));
        setLayoutManager(layoutManager);
    }
}
