package de.motis_project.app2.journey;

import android.support.annotation.Nullable;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;

import butterknife.ButterKnife;

public class JourneyViewHolder extends RecyclerView.ViewHolder {
    final LayoutInflater inflater;

    public JourneyViewHolder(View view, @Nullable LayoutInflater inflater) {
        super(view);
        this.inflater = inflater;
        ButterKnife.bind(this, view);
    }
}