package de.motis_project.app2.ppr;

import android.content.Context;
import android.content.Intent;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import java.util.List;

import de.motis_project.app2.io.Status;
import de.motis_project.app2.ppr.route.RouteWrapper;

public class RouteSummaryAdapter
        extends RecyclerView.Adapter<RouteSummaryViewHolder> {
    private static final String TAG = "RouteSummaryAdapter";

    private final List<RouteWrapper> data;
    private Context context;

    public RouteSummaryAdapter(List<RouteWrapper> data) {
        this.data = data;
    }

    @Override
    public RouteSummaryViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        context = parent.getContext();
        LayoutInflater inflater = LayoutInflater.from(context);
        return new RouteSummaryViewHolder(parent, inflater);
    }

    @Override
    public void onBindViewHolder(RouteSummaryViewHolder holder, int index) {
        if (index < 0 || index >= data.size()) {
            return;
        }

        final RouteWrapper route = data.get(index);
        RouteSummaryViewHolder vh = (RouteSummaryViewHolder) holder;
        vh.setRoute(route, context);
        vh.itemView.setOnClickListener(v -> {
            Status.get().setPprRoute(route);
            Intent intent = new Intent(v.getContext(), RouteDetailActivity.class);
            v.getContext().startActivity(intent);
        });
    }

    @Override
    public int getItemCount() {
        return data.size();
    }

}
