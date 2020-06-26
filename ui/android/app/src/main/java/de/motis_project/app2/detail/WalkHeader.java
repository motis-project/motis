package de.motis_project.app2.detail;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import motis.Connection;
import motis.Walk;

public class WalkHeader implements DetailViewHolder {
    private View layout;

    @BindView(R.id.detail_walk_name)
    TextView walkName;

    WalkHeader(Connection con,
               JourneyUtil.Section section,
               Walk w,
               ViewGroup parent,
               LayoutInflater inflater) {
        layout = inflater.inflate(
                R.layout.detail_walk_header, parent, false);
        ButterKnife.bind(this, layout);

        Context context = inflater.getContext();
        JourneyUtil.tintBackground(context, walkName, JourneyUtil.WALK_CLASS);
        JourneyUtil.setIcon(context, walkName, JourneyUtil.WALK_CLASS);
    }

    @Override
    public View getView() {
        return layout;
    }
}
