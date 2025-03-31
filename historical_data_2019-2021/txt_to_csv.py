import csv

# include deliquency status
def to_csv_performance(input_file, output_file):
    headers = [
        "Loan Sequence Number", "Monthly Reporting Period", "Current Actual UPB", 
        "Current Loan Delinquency Status", "Loan Age", "Remaining Months to Legal Maturity",
        "Defect Settlement Date", "Modification Flag", "Zero Balance Code", 
        "Zero Balance Effective Date", "Current Interest Rate", "Current Deferred UPB", 
        "Due Date of Last Paid Installment (DDLPI)", "MI Recoveries", "Net Sales Proceeds", 
        "Non MI Recoveries", "Expenses", "Legal Costs", "Maintenance and Preservation Costs", 
        "Taxes and Insurance", "Miscellaneous Expenses", "Actual Loss Calculation", 
        "Modification Cost", "Step Modification Flag", "Deferred Payment Plan", 
        "Estimated Loan-to-Value (ELTV)", "Zero Balance Removal UPB", "Delinquent Accrued Interest", 
        "Delinquency Due to Disaster", "Borrower Assistance Status Code", 
        "Current Month Modification Cost", "Interest Bearing UPB"
    ]

    # convert txt to csv and add headers
    with open(input_file, "r") as txt_file, open(output_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        for line in txt_file:
            csv_writer.writerow(line.strip().split("|"))

# include other attributes
def to_csv_origination(input_file, output_file):
    headers = [
        "Credit Score", "First Payment Date", "First Time Homebuyer Flag",
        "Maturity Date", "Metropolitan Statistical Area (MSA) Or Metropolitan Division",
        "Mortgage Insurance Percentage (MI %)", "Number of Units", "Occupancy Status",
        "Original Combined Loan-to-Value (CLTV)", "Original Debt-to-Income (DTI) Ratio",
        "Original UPB", "Original Loan-to-Value (LTV)", "Original Interest Rate", "Channel",
        "Prepayment Penalty Mortgage (PPM) Flag", "Amortization Type (Formerly Product Type)",
        "Property State", "Property Type", "Postal Code", "Loan Sequence Number",
        "Loan Purpose", "Original Loan Term", "Number of Borrowers", "Seller Name",
        "Servicer Name", "Super Conforming Flag", "Pre-HARP Loan Sequence Number",
        "Program Indicator", "HARP Indicator", "Property Valuation Method",
        "Interest Only (I/O) Indicator", "Mortgage Insurance Cancellation Indicator"
    ]

    # convert txt to csv and add headers
    with open(input_file, "r") as txt_file, open(output_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        for line in txt_file:
            csv_writer.writerow(line.strip().split("|"))

input_performance = "raw/historical_data_2021/historical_data_time_2021Q4.txt"
output_performance = "2021/2021Q4_standard_performance.csv"
to_csv_performance(input_performance, output_performance)

input_origination = "raw/historical_data_2021/historical_data_2021Q4.txt"
output_origination = "2021/2021Q4_standard_origination.csv"
to_csv_origination(input_origination, output_origination)