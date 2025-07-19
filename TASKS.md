# Developer Task List

## **Phase 1: Systematic Removal of Hedging Logic**

### **Developer 1: Core Logic and File Deletion**

**Objective**: Remove the core hedging logic and all related files from the project.

**Instructions**:

1.  **Delete the entire `hedge` directory**:
    *   `rm -rf /home/aladhimarkets/Thalex_SimpleQuouter/thalex_py/Thalex_modular/components/hedge/`
2.  **Delete the hedging-related scripts in the root directory**:
    *   `rm /home/aladhimarkets/Thalex_SimpleQuouter/integrate_hedge_manager.py`
    *   `rm /home/aladhimarkets/Thalex_SimpleQuouter/force_hedge_rebalance.py`
    *   `rm /home/aladhimarkets/Thalex_SimpleQuouter/fix_eth_price_fetching.py`
    *   `rm /home/aladhimarkets/Thalex_SimpleQuouter/enable_hedging.py`
3.  **Remove hedging logic from `avellaneda_market_maker.py`**:
    *   Open `/home/aladhimarkets/Thalex_SimpleQuouter/thalex_py/Thalex_modular/components/avellaneda_market_maker.py`.
    *   Remove the `hedge_manager` and `use_hedging` attributes from the `__slots__` definition.
    *   Remove the entire `_initialize_hedge_manager` method.
    *   Remove the `_process_fill_for_hedging` method.
    *   Remove the `report_hedge_status` method.
    *   In the `__init__` method, remove the lines that initialize and start the `hedge_manager`.
    *   In the `handle_order_update` method, remove the call to `_process_fill_for_hedging`.
    *   In the `shutdown` method, remove the code that stops the `hedge_manager`.

---

### **Developer 2: Configuration and Documentation Cleanup**

**Objective**: Remove all references to hedging from configuration files, documentation, and logs.

**Instructions**:

1.  **Remove hedging from `.example.env`**:
    *   Open `/home/aladhimarkets/Thalex_SimpleQuouter/.example.env`.
    *   Delete the entire `# HEDGING CONFIGURATION` section.
2.  **Remove `hedge_state.json` from `.gitignore`**:
    *   Open `/home/aladhimarkets/Thalex_SimpleQuouter/.gitignore`.
    *   Delete the line `hedge_state.json`.
3.  **Remove hedging from `thalex_logging.py`**:
    *   Open `/home/aladhimarkets/Thalex_SimpleQuouter/thalex_py/Thalex_modular/thalex_logging/logger_factory.py`.
    *   Remove the line `'hedge': 'hedge',` from the `LOG_LEVELS` dictionary.
4.  **Remove hedging from documentation**:
    *   Review all `.md` files in the project and remove any sections or references to hedging.

**Once both developers have completed their tasks, we will run the test suite to ensure the bot's core functionality remains intact.**
