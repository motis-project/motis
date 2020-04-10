package de.motis_project.app2.intermodal;

import android.content.Context;
import android.support.annotation.Nullable;
import android.util.AttributeSet;

import de.motis_project.app2.journey.BaseJourneyListView;


public class IntermodalJourneyListView extends BaseJourneyListView<IntermodalQuery> {
    public IntermodalJourneyListView(Context context) {
        super(context);
    }

    public IntermodalJourneyListView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public IntermodalJourneyListView(Context context, @Nullable AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
    }
}
