package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.LocaleList;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import androidx.constraintlayout.core.motion.utils.TypedValues;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.model.LanguageModel;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes3.dex */
public class LanguageUtil {
    private static final String TAG = "LanguageUtil";
    private static LanguageModel[] languageArr;
    private static List<LanguageModel> languageList;

    static {
        LanguageModel[] languageModelArr = {new LanguageModel("de", "Deutsche"), new LanguageModel("en", "English"), new LanguageModel("es", "Español"), new LanguageModel("fr", "Français"), new LanguageModel("it", "Italiaans"), new LanguageModel("ja", "日本語"), new LanguageModel("nl", "Dutch")};
        languageArr = languageModelArr;
        languageList = Arrays.asList(languageModelArr);
    }

    public static Context attachBaseContext(Context context, String str) {
        return updateResources(context, str);
    }

    public static void changeAppLanguage(Context context, String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        if ("jp".equals(str)) {
            str = "ja";
        }
        Resources resources = context.getResources();
        Configuration configuration = resources.getConfiguration();
        configuration.setLocale(new Locale(str));
        resources.updateConfiguration(configuration, resources.getDisplayMetrics());
    }

    /* JADX WARN: Code restructure failed: missing block: B:16:0x002f, code lost:
    
        r0 = "en";
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static String getCurrLanguage(Context context) {
        String language = context.getResources().getConfiguration().locale.getLanguage();
        int i = 0;
        while (true) {
            List<LanguageModel> list = languageList;
            if (list == null || i >= list.size()) {
                break;
            }
            if (language.equals(languageList.get(i).getIso())) {
                break;
            }
            i++;
        }
        String str = (String) SharedPreferencesUtil.get(context, Constans.Language, language);
        return TextUtils.isEmpty(str) ? language : str;
    }

    public static String getCurrLanguageName(String str) {
        int i = 0;
        while (str != null) {
            List<LanguageModel> list = languageList;
            if (list == null || i >= list.size()) {
                return "";
            }
            if (str.equals(languageList.get(i).getIso())) {
                return languageList.get(i).getName();
            }
            i++;
        }
        return "";
    }

    public static String getLanguageByKey(Context context, String str) {
        String str2 = "";
        try {
            JSONObject languageJson = MyApplication.getApplication().getLanguageJson();
            if (languageJson != null) {
                str2 = languageJson.getJSONObject("langs").getString(str);
            }
        } catch (JSONException unused) {
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (!TextUtils.isEmpty(str2)) {
            return str2;
        }
        try {
            return context.getResources().getString(getResId(context, "android_" + str));
        } catch (Resources.NotFoundException unused2) {
            return str;
        }
    }

    public static List<LanguageModel> getLanguageList() {
        return languageList;
    }

    private static int getResId(Context context, String str) {
        return context.getResources().getIdentifier(str, TypedValues.Custom.S_STRING, context.getPackageName());
    }

    public static String getVersionAllLangs(String str) {
        JsonArray jsonArray;
        JsonObject jsonObject = new JsonObject();
        if (!TextUtils.isEmpty(str) && (jsonArray = (JsonArray) new Gson().fromJson(str, JsonArray.class)) != null) {
            for (int i = 0; i < jsonArray.size(); i++) {
                JsonElement jsonElement = jsonArray.get(i);
                if (jsonElement != null && jsonElement.isJsonObject()) {
                    jsonObject.addProperty(jsonElement.getAsJsonObject().get("iso") == null ? "" : jsonElement.getAsJsonObject().get("iso").getAsString(), jsonElement.getAsJsonObject().get("version") != null ? jsonElement.getAsJsonObject().get("version").getAsString() : "");
                }
            }
        }
        return jsonObject.toString();
    }

    public static void setHint(Context context, View view, String str) {
        if (view instanceof EditText) {
            ((EditText) view).setHint(getLanguageByKey(context, str));
        }
    }

    public static void setLanguageList(List<LanguageModel> list) {
        if (list.size() > 0) {
            languageList = list;
        }
    }

    public static void setText(Context context, View view, String str) {
        if (view instanceof TextView) {
            ((TextView) view).setText(getLanguageByKey(context, str));
        }
    }

    private static Context updateResources(Context context, String str) {
        Resources resources = context.getResources();
        if ("jp".equals(str)) {
            str = "ja";
        }
        Locale locale = new Locale(str);
        Configuration configuration = resources.getConfiguration();
        configuration.setLocale(locale);
        configuration.setLocales(new LocaleList(locale));
        return context.createConfigurationContext(configuration);
    }
}
