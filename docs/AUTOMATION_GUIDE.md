# Automation Guide: Windows Task Scheduler Setup

## ⏰ Optimal Timing

### **Recommended Run Time: 6:00 PM ET (18:00 ET)**

**Why this time?**
- ✅ Market closes at 4:00 PM ET
- ✅ Data settles by ~4:30-5:00 PM ET
- ✅ 6:00 PM gives safe buffer for data availability
- ✅ Still early enough to review results same evening

**Alternative times:**
- **Conservative**: 8:00 PM ET (extra safe, guarantees data is available)
- **Late night**: 11:00 PM ET (if you want to review in morning)

**Important:** Adjust for your timezone!
- If you're in PST: 3:00 PM PST = 6:00 PM ET
- If you're in CST: 5:00 PM CST = 6:00 PM ET
- If you're in MST: 4:00 PM MST = 6:00 PM ET

---

## 🪟 Windows Task Scheduler Setup

### **Step 1: Open Task Scheduler**

1. Press `Win + R`
2. Type `taskschd.msc`
3. Press Enter

### **Step 2: Create New Task**

1. Click **"Create Task"** (NOT "Create Basic Task")
2. Name it: `SP500 Paper Trading`
3. Description: `Daily paper trading for sp500-analogs strategy`
4. Select: **"Run whether user is logged on or not"**
5. Check: **"Run with highest privileges"**
6. Configure for: **Windows 10** (or your version)

### **Step 3: Configure Trigger**

1. Go to **"Triggers"** tab
2. Click **"New"**
3. Begin the task: **On a schedule**
4. Settings: **Daily**
5. Start: **6:00:00 PM** (or your chosen time)
6. Recur every: **1 days**
7. Under **Advanced settings**:
   - Check: **Enabled**
   - Stop task if it runs longer than: **1 hour**
   - Check: **Synchronize across time zones** (if you travel)
8. Click **OK**

### **Step 4: Configure Action**

1. Go to **"Actions"** tab
2. Click **"New"**
3. Action: **Start a program**
4. Program/script: `C:\Windows\System32\cmd.exe`
5. Add arguments:
   ```
   /c "cd /d C:\path\to\stocksCorrelation && run_paper_trading.bat"
   ```
   **Replace** `C:\path\to\stocksCorrelation` with your actual path!

6. Start in: `C:\path\to\stocksCorrelation` (same path as above)
7. Click **OK**

### **Step 5: Configure Conditions**

1. Go to **"Conditions"** tab
2. **Power**:
   - Uncheck: "Start the task only if the computer is on AC power"
   - Uncheck: "Stop if the computer switches to battery power"
   - Check: "Wake the computer to run this task"
3. **Network**:
   - Check: "Start only if the following network connection is available"
   - Select: "Any connection"
4. Click **OK**

### **Step 6: Configure Settings**

1. Go to **"Settings"** tab
2. Check: **"Allow task to be run on demand"**
3. Check: **"Run task as soon as possible after a scheduled start is missed"**
4. If the task fails, restart every: **10 minutes**
5. Attempt to restart up to: **3 times**
6. Stop the task if it runs longer than: **1 hour**
7. If the running task does not end when requested: **Stop the task**
8. Click **OK**

### **Step 7: Save and Test**

1. Click **OK** to save the task
2. Enter your Windows password when prompted
3. **Test it now**: Right-click the task → **Run**
4. Check if it works by looking for output in `results/paper_trading/`

---

## 🎯 **Recommended Strategy Path**

### **Phase 1: Grid Search (1-2 days) - OPTIONAL BUT RECOMMENDED**

Find the best parameters on historical data:

```bash
python scripts/gridsearch.py
```

**What it does:**
- Tests many parameter combinations
- Shows which worked best historically
- Gives you top 3-5 candidates

**Review the results in:** `results/experiments/gridsearch_*.csv`

### **Phase 2: Multi-Strategy Paper Trading (1-3 months) - RECOMMENDED**

Test 5 different strategies simultaneously:

#### **Option A: Manual Daily Run**
```bash
python scripts/run_multi_strategy_paper_trading.py
```

#### **Option B: Automated with Task Scheduler**

Modify Task Scheduler **Action** to use:
```
/c "cd /d C:\path\to\stocksCorrelation && .venv\Scripts\activate && python scripts\run_multi_strategy_paper_trading.py"
```

**What it does:**
- Runs 5 different parameter configurations in parallel
- Each gets its own portfolio and tracking
- Generates comparison report showing which performs best

**Strategies tested:**
1. **Conservative**: High threshold (75%), more analogs (30), stricter similarity
2. **Moderate**: Default balanced parameters (70% threshold, 25 analogs)
3. **Aggressive**: Lower threshold (65%), fewer analogs (20), more trades
4. **Spearman Rank**: Uses rank correlation instead of Pearson
5. **Short Window**: 5-day patterns instead of 10-day

**After 1 month:**
- Check `results/paper_trading/strategy_comparison.csv`
- See which strategy has best return %, win rate, profit factor
- **Continue with top 2-3 strategies for another 2 months**

**After 3 months:**
- Pick the best performing strategy
- Use it for live trading (or continue paper trading)

### **Phase 3: Single Strategy Production (ongoing)**

Once you've identified the best strategy:

1. **Update config** (`configs/default.yaml`) with winning parameters
2. **Switch to single strategy** paper trading:
   ```bash
   python scripts/paper_trade_daily.py
   ```
3. **Consider live trading** if:
   - 3 month total return > 15%
   - Win rate consistently > 55%
   - Profit factor > 1.5
   - Max drawdown < 20%

---

## 📊 **Comparison: Single vs Multi-Strategy**

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Single Strategy** | Simple, clean tracking | May not be optimal params | If you trust your config |
| **Multi-Strategy** | Find best params, compare performance | More complex, more data | Optimization & validation |

**My Recommendation:** Start with **Multi-Strategy** for 1-3 months, then switch to **Single Strategy** with the winner.

---

## 🔄 **Daily Workflow (Automated)**

With Task Scheduler set up:

### **Completely Hands-Off:**
1. ✅ Task Scheduler runs automatically at 6:00 PM ET
2. ✅ Fetches prices, updates positions, generates signals
3. ✅ Records everything to CSV
4. ✅ No action needed from you

### **Optional Daily Check (2 minutes):**
1. Open `results/paper_trading/strategy_comparison.csv`
2. Check today's P&L
3. Note any large moves

### **Monthly Review (15 minutes):**
```bash
python scripts/export_to_excel.py
```
1. Open the Excel report
2. Review performance across all strategies
3. Identify trends and patterns

---

## 📈 **What to Track**

### **Weekly (5 minutes):**
- Total return % for each strategy
- Number of trades per strategy
- Any concerning patterns

### **Monthly (30 minutes):**
- Generate Excel report
- Compare strategy performance
- Check win rate and profit factor
- Review largest wins/losses
- Look for parameter patterns

### **Quarterly Decision Point:**
After 3 months:
- **If any strategy is profitable** (>10% total return, >55% win rate):
  → Continue with that strategy, consider live trading
- **If all strategies breakeven** (0-10% return):
  → Try different parameter ranges, continue paper trading
- **If all strategies losing** (<0% return):
  → Strategy may not work in current market conditions, pause or revise approach

---

## 🛠️ **Troubleshooting Task Scheduler**

### **Task doesn't run:**
1. Check Task Scheduler History (View → Show All Running Tasks)
2. Verify the path in "Start in" is correct
3. Try running manually (right-click → Run)

### **Task runs but no output:**
1. Check if virtual environment path is correct
2. Run the bat file manually to see errors
3. Check logs in console output

### **"Access denied" errors:**
1. Make sure "Run with highest privileges" is checked
2. Enter correct Windows password when saving task
3. Try running Task Scheduler as Administrator

---

## 💡 **Pro Tips**

1. **Test first**: Before setting up automation, run manually for a week
2. **Monitor initially**: Check daily for first week to ensure it's working
3. **Keep computer on**: Or ensure "Wake the computer to run this task" is checked
4. **Internet required**: Task needs internet to fetch prices
5. **Backup data**: Keep monthly Excel reports backed up
6. **Review quarterly**: Set calendar reminder to review performance every 3 months

---

## 🎓 **My Recommended Approach**

### **Week 1:**
- Run `python scripts/gridsearch.py` to find good parameters
- Review results, note top 5 parameter combinations
- Start multi-strategy paper trading manually

### **Week 2:**
- Set up Task Scheduler for automation
- Verify it runs correctly for 3 days
- Switch to automated mode

### **Months 1-3:**
- Let it run automatically
- Review weekly (5 min)
- Generate monthly Excel reports
- Track which strategy performs best

### **After Month 3:**
- Analyze results
- Pick winning strategy
- Switch to single-strategy mode
- Consider live trading if profitable

---

## 📞 **Support**

If Task Scheduler isn't working:
1. Check Windows Event Viewer for errors
2. Try running the bat file manually first
3. Verify all paths are absolute (not relative)
4. Ensure Python and virtual environment are working

---

## ✅ **Quick Checklist**

Before automating:
- [ ] Ran paper trading manually successfully
- [ ] Verified data files exist in `data/processed/`
- [ ] Tested multi-strategy script works
- [ ] Noted absolute paths to project directory
- [ ] Verified internet connection is stable
- [ ] Decided on run time (recommend 6:00 PM ET)
- [ ] Set up Task Scheduler with correct paths
- [ ] Tested task runs manually
- [ ] Verified output files are created
- [ ] Set up monthly calendar reminder for review

You're ready to automate! 🚀
