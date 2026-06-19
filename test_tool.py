from tools.emergency_checker import check_emergency_severity

result = check_emergency_severity.invoke(
    "I have chest pain and difficulty breathing"
)

print(result)