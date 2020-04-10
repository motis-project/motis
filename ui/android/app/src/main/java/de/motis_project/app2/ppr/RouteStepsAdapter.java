package de.motis_project.app2.ppr;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import java.util.List;

import de.motis_project.app2.ppr.route.RouteWrapper;
import de.motis_project.app2.ppr.route.StepInfo;

public class RouteStepsAdapter
        extends RecyclerView.Adapter<RouteStepViewHolder> {
    private static final String TAG = "RouteStepsAdapter";

    private final List<StepInfo> steps;
    private Context context;
    private OnClickListener clickListener;

    public RouteStepsAdapter(RouteWrapper route) {
        this.steps = route.getSteps();
    }

    @Override
    public RouteStepViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        context = parent.getContext();
        LayoutInflater inflater = LayoutInflater.from(context);
        return new RouteStepViewHolder(parent, inflater);
    }

    @Override
    public void onBindViewHolder(RouteStepViewHolder holder, int index) {
        if (index < 0 || index >= steps.size()) {
            return;
        }

        final StepInfo step = steps.get(index);
        RouteStepViewHolder vh = (RouteStepViewHolder) holder;
        vh.setStep(step, context);
        vh.itemView.setOnClickListener(v -> {
            if (clickListener != null) {
                clickListener.onStepClicked(step);
            }
        });
    }

    @Override
    public int getItemCount() {
        return steps.size();
    }

    public void setClickListener(OnClickListener clickListener) {
        this.clickListener = clickListener;
    }

    public interface OnClickListener {
        public void onStepClicked(StepInfo step);
    }
}
