package de.motis_project.app2.detail;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.ppr.route.StepInfo;

public class WalkStep implements DetailViewHolder {
    private View layout;

    @BindView(R.id.detail_step_name)
    TextView description;

    @BindView(R.id.detail_step_time)
    TextView stopTime;

    @BindView(R.id.detail_step_vertline)
    View line;

    @BindView(R.id.detail_step_bullet)
    View bullet;

    protected final StepInfo stepInfo;
    protected final DetailClickHandler clickHandler;

    WalkStep(StepInfo stepInfo, long time, ViewGroup parent,
             LayoutInflater inflater) {
        this.stepInfo = stepInfo;
        clickHandler = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_walk_step, parent, false);
        ButterKnife.bind(this, layout);

        Context context = inflater.getContext();
        JourneyUtil.setBackgroundColor(context, line, JourneyUtil.WALK_CLASS);
        JourneyUtil.tintBackground(context, bullet, JourneyUtil.WALK_CLASS);

        stopTime.setText(TimeUtil.formatTime(time));
        stopTime.setCompoundDrawablesWithIntrinsicBounds(0, 0, stepInfo.getIcon(), 0);
        description.setText(stepInfo.getText(), stepInfo.getTextBufferType());
    }

    @Override
    public View getView() {
        return layout;
    }

    @OnClick(R.id.detail_step)
    public void onClick() {
        clickHandler.walkStepClicked(stepInfo);
    }
}
