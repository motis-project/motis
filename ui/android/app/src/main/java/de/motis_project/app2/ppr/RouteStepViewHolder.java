package de.motis_project.app2.ppr;

import android.content.Context;
import android.support.v7.widget.AppCompatImageView;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.R;
import de.motis_project.app2.ppr.route.StepInfo;


public class RouteStepViewHolder extends RecyclerView.ViewHolder {
    private static final String TAG = "RouteStepViewHolder";
    final LayoutInflater inflater;

    @BindView(R.id.step_icon)
    AppCompatImageView iconView;

    @BindView(R.id.description)
    TextView description;

    @BindView(R.id.distance)
    TextView distance;

    public RouteStepViewHolder(ViewGroup parent, LayoutInflater inflater) {
        super(inflater.inflate(R.layout.ppr_route_step, parent, false));
        this.inflater = inflater;
        ButterKnife.bind(this, this.itemView);
    }

    void setStep(StepInfo step, Context context) {
        description.setText(step.getText(), step.getTextBufferType());
        distance.setText(Format.formatDistance(step.getDistance()));
        iconView.setImageResource(step.getIcon());
    }
}
