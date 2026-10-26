package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.BlackModel;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class BlackAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<BlackModel.User> mList;
    private OnSelectedChangeListener onSelectedChangeListener;

    public interface OnSelectedChangeListener {
        void onSelectedChange(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgGender;
        private CircleImageView mImgHead;
        private ImageView mImgSelected;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.item_black_img_head);
            this.mTxtName = (TextView) view.findViewById(R.id.item_black_txt_name);
            this.mImgGender = (ImageView) view.findViewById(R.id.item_black_img_gender);
            this.mImgSelected = (ImageView) view.findViewById(R.id.item_black_img_selected);
        }
    }

    public BlackAdapter(Context context, List<BlackModel.User> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mImgHead);
        viewHolder.mImgHead.setKing(this.mList.get(i).getKing(), this.mContext, this.mList.get(i).getGradeIcon());
        viewHolder.mTxtName.setText(this.mList.get(i).getName());
        if (this.mList.get(i).getChecked() == 1) {
            viewHolder.mImgSelected.setImageResource(R.mipmap.check_checked);
        } else {
            viewHolder.mImgSelected.setImageResource(R.mipmap.check_normal);
        }
        if (this.mList.get(i).getGender() == 2) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mList.get(i).getGender() == 1) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        viewHolder.mImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BlackAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                IntentUtil.startUserInfoActivity(BlackAdapter.this.mContext, ((BlackModel.User) BlackAdapter.this.mList.get(i)).getUserId());
            }
        });
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BlackAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                int checked = ((BlackModel.User) BlackAdapter.this.mList.get(i)).getChecked();
                BlackAdapter blackAdapter = BlackAdapter.this;
                int i2 = 0;
                if (checked == 1) {
                    ((BlackModel.User) blackAdapter.mList.get(i)).setChecked(0);
                    viewHolder.mImgSelected.setImageResource(R.mipmap.check_normal);
                } else {
                    ((BlackModel.User) blackAdapter.mList.get(i)).setChecked(1);
                    viewHolder.mImgSelected.setImageResource(R.mipmap.check_checked);
                }
                if (BlackAdapter.this.onSelectedChangeListener == null) {
                    return;
                }
                int i3 = 0;
                while (true) {
                    int size = BlackAdapter.this.mList.size();
                    BlackAdapter blackAdapter2 = BlackAdapter.this;
                    if (i2 >= size) {
                        blackAdapter2.onSelectedChangeListener.onSelectedChange(i3);
                        return;
                    } else {
                        if (((BlackModel.User) blackAdapter2.mList.get(i2)).getChecked() == 1) {
                            i3++;
                        }
                        i2++;
                    }
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_black, viewGroup, false));
    }

    public void setOnSelectedChangeListener(OnSelectedChangeListener onSelectedChangeListener) {
        this.onSelectedChangeListener = onSelectedChangeListener;
    }
}
