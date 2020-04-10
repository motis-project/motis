package de.motis_project.app2.ppr;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.support.v4.graphics.drawable.DrawableCompat;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.ppr.route.RouteWrapper;
import motis.ppr.Route;


public class RouteSummaryViewHolder extends RecyclerView.ViewHolder {
    private static final String TAG = "RouteSummaryViewHolder";
    final LayoutInflater inflater;

    @BindView(R.id.dot)
    View dotView;

    @BindView(R.id.duration)
    TextView duration;

    @BindView(R.id.distance)
    TextView distance;

    @BindView(R.id.accessibility)
    TextView accessibility;

    @BindView(R.id.elevation_up)
    TextView elevationUp;

    @BindView(R.id.elevation_down)
    TextView elevationDown;

    public RouteSummaryViewHolder(ViewGroup parent, LayoutInflater inflater) {
        super(inflater.inflate(R.layout.ppr_list_item, parent, false));
        this.inflater = inflater;
        ButterKnife.bind(this, this.itemView);
    }

    void setRoute(RouteWrapper route, Context context) {
        Route r = route.getRoute();
        duration.setText(TimeUtil.formatDuration(r.duration()));
        distance.setText(Format.formatDistance(r.distance()));
        accessibility.setText(Integer.toString(r.accessibility()));
        elevationUp.setText("↑ " + Format.formatDistance(r.elevationUp()));
        elevationDown.setText("↓ " + Format.formatDistance(r.elevationDown()));

        int routeColor = route.getColor(context);

        Drawable bg = DrawableCompat.wrap(dotView.getBackground());
        DrawableCompat.setTint(bg.mutate(), routeColor);
        dotView.setBackground(bg);
    }
}
