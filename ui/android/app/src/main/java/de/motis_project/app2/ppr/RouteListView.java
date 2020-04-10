package de.motis_project.app2.ppr;

import android.content.Context;
import android.support.annotation.Nullable;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.AttributeSet;

import de.motis_project.app2.lib.SimpleDividerItemDecoration;


public class RouteListView extends RecyclerView {
    private static final String TAG = "RouteListView";

    private final LinearLayoutManager layoutManager = new LinearLayoutManager(getContext());

    public RouteListView(Context context) {
        super(context);
        init();
    }

    public RouteListView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public RouteListView(Context context, @Nullable AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init();
    }

    private void init() {
        addItemDecoration(new SimpleDividerItemDecoration(getContext()));
        setLayoutManager(layoutManager);
    }
}
